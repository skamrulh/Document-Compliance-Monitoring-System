"""API integration tests — all infrastructure mocked via conftest.py stubs."""
import sys, os, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from ml_pipeline.models import ComplianceNERModel, RiskClassifier, SemanticSearchEngine
from data_ingestion.pipeline import DocumentIngestor
import app.main as main_mod

# Inject state without running lifespan (Redis/Kafka not needed)
main_mod.redis_client = None
main_mod.kafka_producer = None
main_mod.document_store.clear()
main_mod.app.state.ner_model         = ComplianceNERModel()
main_mod.app.state.risk_classifier   = RiskClassifier()
main_mod.app.state.search_engine     = SemanticSearchEngine()
main_mod.app.state.document_ingestor = DocumentIngestor({})

client = TestClient(main_mod.app, raise_server_exceptions=False)
H = {"X-API-Key": "demo"}

def test_health_200():
    r = client.get("/health"); assert r.status_code == 200
    d = r.json(); assert d["status"]=="healthy" and "timestamp" in d

def test_metrics_prometheus():
    r = client.get("/metrics"); assert r.status_code == 200
    assert "HELP" in r.text or "http_requests" in r.text

def test_no_key_403():
    assert client.post("/api/v1/search", json={"text":"q"}).status_code == 403

def test_wrong_key_403():
    assert client.post("/api/v1/search",json={"text":"q"},headers={"X-API-Key":"nope"}).status_code == 403

def test_demo_key_accepted():
    assert client.post("/api/v1/search", json={"text":"q"}, headers=H).status_code == 200

def test_403_detail_mentions_header():
    r = client.post("/api/v1/search", json={"text":"q"})
    assert "X-API-Key" in r.json()["detail"]

def test_search_empty_index():
    r = client.post("/api/v1/search",json={"text":"compliance"},headers=H)
    d = r.json(); assert d["results"]==[] and d["total_results"]==0

def test_search_schema():
    r = client.post("/api/v1/search",json={"text":"GDPR","limit":5},headers=H)
    for f in ("query","results","total_results","index_size","search_type"):
        assert f in r.json()

def test_search_invalid_type_422():
    assert client.post("/api/v1/search",json={"text":"q","search_type":"mindreading"},headers=H).status_code==422

def test_get_doc_404():
    assert client.get("/api/v1/documents/no-such-doc",headers=H).status_code==404

def test_analyze_404():
    assert client.post("/api/v1/documents/ghost/analyze",headers=H).status_code==404

def test_batch_unknown_not_found():
    r = client.post("/api/v1/documents/batch",json={"document_ids":["x","y"],"priority":"normal"},headers=H)
    assert r.status_code==200
    assert all(i["status"]=="not_found" for i in r.json()["results"])

def test_batch_invalid_priority_422():
    assert client.post("/api/v1/documents/batch",json={"document_ids":["a"],"priority":"ludicrous"},headers=H).status_code==422

def test_batch_has_message():
    r = client.post("/api/v1/documents/batch",json={"document_ids":["z"],"priority":"high"},headers=H)
    assert "message" in r.json()

def test_compliance_unknown_404():
    assert client.post("/api/v1/compliance/check",json={"document_id":"nobody"},headers=H).status_code==404

def test_compliance_full_pipeline():
    doc_id = "test-compliance-001"
    main_mod.document_store[doc_id] = {
        "id":doc_id,"filename":"contract.pdf","file_type":"pdf",
        "text":("This contract between Acme Corp and Beta Ltd is subject to GDPR compliance. "
                "A breach of this agreement may result in penalty and lawsuit. SOX audit violation."),
        "metadata":{"file_type":"pdf"},"uploaded_at":"2024-01-01T00:00:00",
        "entities":None,"risk":None,"analyzed_at":None,
    }
    r = client.post("/api/v1/compliance/check",
                    json={"document_id":doc_id,"regulations":["GDPR","SOX"]},headers=H)
    assert r.status_code==200
    d = r.json()
    assert d["compliance_status"] in ("PASS","FAIL")
    assert "risk_assessment" in d and "entities_detected" in d
    assert "GDPR" in d["regulation_coverage"] and "SOX" in d["regulation_coverage"]

def test_risk_deterministic():
    from ml_pipeline.models import RiskClassifier
    clf = RiskClassifier(); text = "breach violation penalty lawsuit GDPR SOX"
    r1 = clf.classify_risk(text,[]); r2 = clf.classify_risk(text,[])
    assert r1["risk_level"]==r2["risk_level"] and r1["score"]==r2["score"]
