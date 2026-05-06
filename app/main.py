"""FastAPI — Intelligent Document Processing & Compliance System v2.0"""
import asyncio, json, logging, os, time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiofiles
import redis.asyncio as redis
from aiokafka import AIOKafkaProducer
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from data_ingestion.pipeline import DocumentIngestor
from ml_pipeline.models import ComplianceNERModel, RiskClassifier, SemanticSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUEST_COUNT    = Counter("http_requests_total", "HTTP requests", ["method","endpoint","status_code"])
REQUEST_LATENCY  = Histogram("http_request_duration_seconds", "Latency", ["endpoint"])
DOCS_PROCESSED   = Counter("documents_processed_total", "Documents processed", ["doc_type","risk_level"])

document_store: Dict[str, Dict[str, Any]] = {}

API_KEY_NAME  = "X-API-Key"
_api_key_hdr  = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
_ALLOWED_KEYS = set(os.getenv("ALLOWED_API_KEYS", "demo,changeme-in-production").split(","))


async def verify_api_key(api_key: Optional[str] = Depends(_api_key_hdr)) -> str:
    if api_key is None or api_key.strip() not in _ALLOWED_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key. Use header: X-API-Key: demo")
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, kafka_producer
    try:
        redis_client = redis.Redis(host=os.getenv("REDIS_HOST","localhost"), port=int(os.getenv("REDIS_PORT",6379)), decode_responses=True)
        await redis_client.ping(); logger.info("Redis connected.")
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}"); redis_client = None
    try:
        kafka_producer = AIOKafkaProducer(bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS","localhost:9092"))
        await kafka_producer.start(); logger.info("Kafka connected.")
    except Exception as e:
        logger.warning(f"Kafka unavailable: {e}"); kafka_producer = None
    logger.info("Loading ML models…")
    app.state.ner_model         = ComplianceNERModel()
    app.state.risk_classifier   = RiskClassifier()
    app.state.search_engine     = SemanticSearchEngine()
    app.state.document_ingestor = DocumentIngestor({})
    logger.info("ML models ready.")
    yield
    if redis_client: await redis_client.close()
    if kafka_producer: await kafka_producer.stop()


redis_client = kafka_producer = None
_CORS = os.getenv("CORS_ORIGINS","*").split(",")

app = FastAPI(title="Intelligent Document Processing & Compliance API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=_CORS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    t = time.time()
    resp = await call_next(request)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=resp.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.time()-t)
    return resp


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    search_type: str = Field("semantic", pattern="^(semantic|keyword|hybrid)$")
    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(10, ge=1, le=100)

class ComplianceCheckRequest(BaseModel):
    document_id: str
    regulations: List[str] = Field(default=["SOX","GDPR","PCI-DSS"])

class BatchProcessRequest(BaseModel):
    document_ids: List[str] = Field(..., min_length=1)
    priority: str = Field("normal", pattern="^(low|normal|high|critical)$")


# ── Background ML task ────────────────────────────────────────────────────────
def _analyze_and_index(doc_id, text, metadata, ner, risk, search):
    try:
        ents = ner.extract_entities(text)
        r    = risk.classify_risk(text, ents["transformer_entities"], ents.get("doc_type","general"))
        document_store[doc_id].update({"entities": ents, "risk": r, "analyzed_at": datetime.utcnow().isoformat()})
        search.add_document(doc_id, text, metadata)
        DOCS_PROCESSED.labels(doc_type=ents.get("doc_type","unknown"), risk_level=r.get("risk_level","UNKNOWN")).inc()
    except Exception as e:
        logger.error(f"Background analysis failed for {doc_id}: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status":"healthy","timestamp":datetime.utcnow().isoformat(),"service":"compliance-monitoring",
            "models_loaded": hasattr(app.state,"ner_model"),
            "redis_connected": redis_client is not None,
            "kafka_connected": kafka_producer is not None,
            "documents_indexed": app.state.search_engine.index_size if hasattr(app.state,"search_engine") else 0}

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/v1/documents/upload", status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), _k: str = Depends(verify_api_key)):
    """Upload document for processing. Add header: X-API-Key: demo"""
    try:
        content = await file.read()
        tmp = f"/tmp/{file.filename}"
        async with aiofiles.open(tmp, "wb") as f: await f.write(content)
        raw = app.state.document_ingestor.ingest_document(tmp)
        text = raw.content.decode("utf-8", errors="replace") if raw.content else ""
        document_store[raw.id] = {"id":raw.id,"filename":raw.filename,"file_type":raw.file_type,
            "text":text,"metadata":raw.metadata,"uploaded_at":raw.upload_timestamp.isoformat(),
            "entities":None,"risk":None,"analyzed_at":None}
        background_tasks.add_task(_analyze_and_index, raw.id, text, raw.metadata,
            app.state.ner_model, app.state.risk_classifier, app.state.search_engine)
        if kafka_producer:
            await kafka_producer.send_and_wait("document-processing",
                json.dumps({"document_id":raw.id,"filename":raw.filename,"timestamp":datetime.utcnow().isoformat()}).encode())
        if os.path.exists(tmp): os.remove(tmp)
        return {"document_id":raw.id,"status":"processing",
                "message":"Document queued. Poll /api/v1/documents/{id} for results.","filename":raw.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/documents/{doc_id}")
async def get_document(doc_id: str, _k: str = Depends(verify_api_key)):
    """Retrieve document info and analysis results."""
    doc = document_store.get(doc_id)
    if not doc: raise HTTPException(404, f"Document '{doc_id}' not found.")
    return doc

@app.post("/api/v1/documents/{doc_id}/analyze")
async def analyze_document(doc_id: str, _k: str = Depends(verify_api_key)):
    """Run (or re-run) NER + risk classification on an uploaded document."""
    doc = document_store.get(doc_id)
    if not doc: raise HTTPException(404, f"Document '{doc_id}' not found.")
    text = doc.get("text","")
    if not text: raise HTTPException(422, "No text content available.")
    ents = app.state.ner_model.extract_entities(text)
    risk = app.state.risk_classifier.classify_risk(text, ents["transformer_entities"], ents.get("doc_type","general"))
    document_store[doc_id].update({"entities":ents,"risk":risk,"analyzed_at":datetime.utcnow().isoformat()})
    return {"document_id":doc_id,"filename":doc.get("filename"),"entities":ents,
            "risk_assessment":risk,"analyzed_at":document_store[doc_id]["analyzed_at"]}

@app.post("/api/v1/documents/batch")
async def batch_process(req: BatchProcessRequest, background_tasks: BackgroundTasks, _k: str = Depends(verify_api_key)):
    """Submit batch of uploaded document IDs for (re-)analysis."""
    results = []
    for did in req.document_ids:
        doc = document_store.get(did)
        if not doc: results.append({"document_id":did,"status":"not_found"}); continue
        background_tasks.add_task(_analyze_and_index, did, doc.get("text",""), doc.get("metadata",{}),
            app.state.ner_model, app.state.risk_classifier, app.state.search_engine)
        results.append({"document_id":did,"status":"queued","priority":req.priority})
    return {"batch_size":len(req.document_ids),"results":results,
            "message":"Batch analysis queued. Poll individual documents for results."}

@app.post("/api/v1/search")
async def semantic_search(query: QueryRequest, _k: str = Depends(verify_api_key)):
    """Semantic search across indexed documents using sentence-transformer embeddings."""
    results = app.state.search_engine.search_similar(query.text, k=query.limit)
    return {"query":query.text,"search_type":query.search_type,"results":results,
            "total_results":len(results),"index_size":app.state.search_engine.index_size}

@app.post("/api/v1/compliance/check")
async def compliance_check(req: ComplianceCheckRequest, _k: str = Depends(verify_api_key)):
    """Full compliance report: NER + risk scoring + regulation coverage."""
    doc = document_store.get(req.document_id)
    if not doc: raise HTTPException(404, f"Document '{req.document_id}' not found. Upload it first.")
    text = doc.get("text","")
    if doc.get("entities") is None:
        ents = app.state.ner_model.extract_entities(text)
        risk = app.state.risk_classifier.classify_risk(text, ents["transformer_entities"], ents.get("doc_type","general"))
        document_store[req.document_id].update({"entities":ents,"risk":risk})
    else:
        ents, risk = doc["entities"], doc["risk"]
    detected = [e["word"] for e in ents["transformer_entities"] if e.get("entity")=="REGULATION"]
    reg_cov   = {r: any(r.upper()==d.upper() for d in detected) for r in req.regulations}
    return {"document_id":req.document_id,"filename":doc.get("filename"),
            "compliance_status":"FAIL" if risk["risk_level"]=="HIGH" else "PASS",
            "risk_assessment":risk,"entities_detected":ents,
            "regulation_coverage":reg_cov,"checked_regulations":req.regulations,
            "timestamp":datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn; uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
