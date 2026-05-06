# Intelligent Document Processing & Compliance System

![CI](https://github.com/<your-username>/Intelligent-Document-Processing-System/actions/workflows/main.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Transformers](https://img.shields.io/badge/NLP-HuggingFace_Transformers-yellow)
![Docker](https://img.shields.io/badge/deploy-Docker-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade distributed ML system that ingests messy real-world documents (PDFs, scanned images, contracts), extracts named entities with fine-tuned BERT, scores compliance risk deterministically, and enables semantic retrieval across the document corpus.

---

## Business Context

Compliance teams at regulated firms review thousands of documents for regulatory violations — GDPR, SOX, HIPAA, PCI-DSS. Manual review is slow and error-prone. This system automates three core questions:

- **What entities are in this document?** → `dslim/bert-base-NER` transformer model
- **What is the compliance risk level?** → Deterministic, auditable rule-based scoring
- **Which documents match this query?** → Semantic search with sentence-transformer embeddings

---

## Architecture

```
Client (curl / frontend)  →  X-API-Key: demo
         │
  ┌──────▼───────────────────────────────────────────────┐
  │  FastAPI  (async, 4 workers)          port 8000      │
  │  ├── API key auth   X-API-Key header                 │
  │  ├── Rate limiting  via Redis                        │
  │  └── Prometheus metrics  /metrics                    │
  └──────┬───────────────────────────────────────────────┘
         │
  ┌──────▼─────────────────┐
  │  Data Ingestion         │
  │  pdfplumber  ← digital  │
  │  PyMuPDF+OCR ← scanned  │
  │  pytesseract ← images   │
  └──────┬─────────────────┘
         │ extracted text
  ┌──────▼──────────────────────────────────────────────┐
  │  ML Pipeline                                        │
  │  ComplianceNERModel                                 │
  │    dslim/bert-base-NER (HuggingFace Transformers)  │
  │    + regex regulation patterns (GDPR, SOX, …)      │
  │                  ↓ entities                         │
  │  RiskClassifier                                     │
  │    keyword density × entity signals × doc-type     │
  │    Deterministic, auditable, zero randomness        │
  │                  ↓ risk score + flags               │
  │  SemanticSearchEngine                               │
  │    all-MiniLM-L6-v2 (SentenceTransformers)         │
  │    cosine-similarity over in-memory embedding index │
  └─────────────────────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────────────┐
  │  Infrastructure                                     │
  │  Kafka   → async event queue for processing         │
  │  Redis   → rate-limit counters + caching            │
  │  MongoDB → persistent document store                │
  │  Prometheus + Grafana → observability               │
  └─────────────────────────────────────────────────────┘
```

---

## ML Models

| Component | Model / Method | Purpose |
|---|---|---|
| NER | `dslim/bert-base-NER` | Extract PER, ORG, LOC entities |
| Regulation Detection | Regex patterns | Detect GDPR, SOX, HIPAA, PCI-DSS, CCPA, NIST… |
| Risk Scoring | Rule-based (keyword + entity signals) | Deterministic compliance risk — HIGH/MEDIUM/LOW |
| Semantic Search | `all-MiniLM-L6-v2` | Cosine-similarity document retrieval |

---

## Project Structure

```
├── app/
│   └── main.py                  # FastAPI — all endpoints, auth, Prometheus metrics
├── ml_pipeline/
│   └── models.py                # ComplianceNERModel, RiskClassifier, SemanticSearchEngine
├── data_ingestion/
│   └── pipeline.py              # DocumentIngestor — PDF, OCR, image processing
├── tests/
│   ├── conftest.py              # Stubs heavy ML libs for fast CI (no GPU needed)
│   ├── test_models.py           # 21 unit tests for all ML model classes
│   └── test_api.py              # 17 API integration tests
├── .github/workflows/main.yml   # CI: lint → test → Docker build & push (fork-safe)
├── Dockerfile                   # Non-root user, system-level OCR deps
├── docker-compose.yaml          # API, Redis, Kafka, Mongo, Prometheus, Grafana, MLflow
├── prometheus.yml               # Prometheus scrape config
├── alerts.yml                   # 4 alert rules (error rate, latency, downtime, risk spike)
└── deployment.yaml              # Kubernetes Deployment + LoadBalancer Service
```

---

## Quick Start

### Docker Compose (full stack)

```bash
git clone https://github.com/<your-username>/Intelligent-Document-Processing-System.git
cd Intelligent-Document-Processing-System
docker compose up --build
```

| Service | URL |
|---|---|
| API + Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |
| MLflow | http://localhost:5000 |

### Local (Python only)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

---

## API Usage

All endpoints require header: **`X-API-Key: demo`**

### 1. Upload a document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "X-API-Key: demo" \
  -F "file=@contract.pdf"
# → {"document_id": "doc_a3f9c2b1d4e87654", "status": "processing", ...}
```

### 2. Get analysis results
```bash
curl http://localhost:8000/api/v1/documents/doc_a3f9c2b1d4e87654 \
  -H "X-API-Key: demo"
```

### 3. Compliance check
```bash
curl -X POST http://localhost:8000/api/v1/compliance/check \
  -H "X-API-Key: demo" -H "Content-Type: application/json" \
  -d '{"document_id": "doc_a3f9c2b1d4e87654", "regulations": ["GDPR","SOX"]}'
```
```json
{
  "compliance_status": "FAIL",
  "risk_assessment": {
    "risk_level": "HIGH", "confidence": 0.94, "score": 0.72,
    "flags": ["High-risk terms: breach, violation, penalty"]
  },
  "entities_detected": {
    "transformer_entities": [
      {"entity": "ORG",        "word": "Acme Corp", "score": 0.98},
      {"entity": "REGULATION", "word": "GDPR",      "score": 1.0}
    ],
    "doc_type": "contract"
  },
  "regulation_coverage": {"GDPR": true, "SOX": true}
}
```

### 4. Semantic search
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: demo" -H "Content-Type: application/json" \
  -d '{"text": "GDPR data breach notification", "limit": 5}'
```

### 5. Batch processing
```bash
curl -X POST http://localhost:8000/api/v1/documents/batch \
  -H "X-API-Key: demo" -H "Content-Type: application/json" \
  -d '{"document_ids": ["doc_abc123", "doc_def456"], "priority": "high"}'
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis hostname |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker |
| `ALLOWED_API_KEYS` | `demo,changeme-in-production` | Comma-separated valid API keys |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |

---

## Running Tests

```bash
pytest tests/ -v
# 38 passed in ~2s  (ML models mocked — no GPU required)
```

---

## Observability

`GET /metrics` exposes Prometheus metrics. Pre-configured alerts in `alerts.yml`:
- **HighErrorRate** — > 5% 5xx responses over 5 min
- **SlowResponseTime** — p95 latency > 2s
- **APIDown** — service unreachable for > 1 min
- **HighRiskDocumentSpike** — surge in HIGH-risk documents

---

## Tech Stack

`Python 3.11` · `FastAPI` · `HuggingFace Transformers` · `Sentence-Transformers` · `Apache Kafka` · `Redis` · `MongoDB` · `Prometheus` · `Grafana` · `Docker` · `Kubernetes` · `GitHub Actions` · `pytest`
