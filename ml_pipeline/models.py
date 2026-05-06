"""
ML pipeline models for Intelligent Document Processing & Compliance System.

Three production components:
  ComplianceNERModel   — HuggingFace transformer NER (dslim/bert-base-NER)
  RiskClassifier       — Deterministic rule-based scoring (auditable, no randomness)
  SemanticSearchEngine — Sentence-transformer embeddings with cosine-similarity retrieval
"""

import logging
import re
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIGH_RISK_KEYWORDS: List[str] = [
    "breach", "violation", "penalty", "fine", "lawsuit", "litigation",
    "non-compliance", "noncompliance", "fraud", "misconduct", "sanction",
    "injunction", "criminal", "terminate", "revoke", "suspend",
]
MEDIUM_RISK_KEYWORDS: List[str] = [
    "audit", "investigation", "review", "warning", "notice", "remediation",
    "corrective action", "deficiency", "gap", "risk", "exposure",
    "regulatory", "inspection", "inquiry",
]
REGULATION_PATTERNS: List[str] = [
    r"\bGDPR\b", r"\bHIPAA\b", r"\bSOX\b", r"\bPCI[\s-]?DSS\b",
    r"\bFERPA\b", r"\bCCPA\b", r"\bISO[\s-]?27001\b", r"\bNIST\b",
    r"\bFINRA\b", r"\bSEC\b",
]

# ---------------------------------------------------------------------------
# Lazy model loaders
# ---------------------------------------------------------------------------
_ner_pipeline = None
_embedding_model = None


def _get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading NER model (dslim/bert-base-NER)…")
            _ner_pipeline = hf_pipeline(
                "ner", model="dslim/bert-base-NER",
                aggregation_strategy="simple", device=-1,
            )
            logger.info("NER model loaded.")
        except Exception as e:
            logger.error(f"NER model load failed: {e}")
            _ner_pipeline = None
    return _ner_pipeline


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model (all-MiniLM-L6-v2)…")
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(f"Embedding model load failed: {e}")
            _embedding_model = None
    return _embedding_model


# ---------------------------------------------------------------------------
# ComplianceNERModel
# ---------------------------------------------------------------------------
class ComplianceNERModel:
    """
    Named Entity Recognition for compliance documents.
    Uses dslim/bert-base-NER + regex regulation patterns.
    Falls back to regex-only when transformer weights are unavailable.
    """
    _ENTITY_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION", "MISC": "MISC"}

    def extract_entities(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"transformer_entities": [], "doc_type": "unknown", "entity_count": 0}

        entities: List[Dict] = []

        # Transformer NER
        ner = _get_ner_pipeline()
        if ner is not None:
            chunk_size, seen = 1000, set()
            for chunk in [text[i:i+chunk_size] for i in range(0, min(len(text), 5000), chunk_size)]:
                try:
                    for ent in ner(chunk):
                        word  = ent.get("word", "").strip()
                        label = self._ENTITY_MAP.get(ent.get("entity_group", ""), ent.get("entity_group", ""))
                        score = float(ent.get("score", 0.0))
                        key   = (word.lower(), label)
                        if key not in seen and len(word) > 1:
                            seen.add(key)
                            entities.append({"entity": label, "word": word, "score": round(score, 4), "source": "transformer"})
                except Exception as e:
                    logger.warning(f"NER chunk error: {e}")
        else:
            logger.warning("NER model unavailable — regex fallback active.")
            for m in re.finditer(r"\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+\b", text):
                w = m.group()
                if not any(e["word"] == w for e in entities):
                    entities.append({"entity": "ORG", "word": w, "score": 0.6, "source": "regex"})

        # Regulation pattern matching
        for pattern in REGULATION_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                word = m.group().upper()
                if not any(e["word"] == word and e["entity"] == "REGULATION" for e in entities):
                    entities.append({"entity": "REGULATION", "word": word, "score": 1.0, "source": "pattern"})

        # Doc type
        tl = text.lower()
        if any(w in tl for w in ["agreement", "contract", "parties", "whereas", "hereinafter"]):
            doc_type = "contract"
        elif any(w in tl for w in ["invoice", "payment", "amount due", "receipt"]):
            doc_type = "financial"
        elif any(w in tl for w in ["audit", "finding", "recommendation", "scope"]):
            doc_type = "audit_report"
        elif any(w in tl for w in ["policy", "procedure", "guideline", "compliance"]):
            doc_type = "policy"
        else:
            doc_type = "general"

        return {"transformer_entities": entities, "doc_type": doc_type, "entity_count": len(entities)}


# ---------------------------------------------------------------------------
# RiskClassifier
# ---------------------------------------------------------------------------
class RiskClassifier:
    """
    Deterministic, auditable rule-based risk scoring.
    Combines keyword density, entity signals, and doc-type multipliers.
    No randomness — same input always produces same output.
    """
    _DOC_MULTIPLIERS = {
        "contract": 1.2, "financial": 1.3, "audit_report": 1.5, "policy": 0.9, "general": 1.0,
    }

    def classify_risk(self, text: str, entities: List[Dict], doc_type: str = "general") -> Dict[str, Any]:
        if not text:
            return {"risk_level": "LOW", "confidence": 0.5, "score": 0.0, "flags": ["No content available"]}

        tl, flags, score = text.lower(), [], 0.0

        high_hits = [kw for kw in HIGH_RISK_KEYWORDS if kw in tl]
        med_hits  = [kw for kw in MEDIUM_RISK_KEYWORDS if kw in tl]
        score += len(high_hits) * 0.15 + len(med_hits) * 0.06

        if high_hits: flags.append(f"High-risk terms: {', '.join(high_hits[:5])}")
        if med_hits:  flags.append(f"Medium-risk terms: {', '.join(med_hits[:5])}")

        regs = [e for e in entities if e.get("entity") == "REGULATION"]
        orgs = [e for e in entities if e.get("entity") == "ORG"]
        if len(regs) >= 3:
            score += 0.20
            flags.append(f"Multiple regulations cited: {[r['word'] for r in regs]}")
        elif regs:
            score += 0.08
        if not orgs and doc_type in ("contract", "financial"):
            score += 0.10
            flags.append("No org entities in financial/contract document")

        score = min(1.0, score * self._DOC_MULTIPLIERS.get(doc_type, 1.0))

        if score >= 0.45:   risk_level, conf = "HIGH",   round(min(0.99, 0.70 + score * 0.25), 3)
        elif score >= 0.20: risk_level, conf = "MEDIUM", round(min(0.99, 0.60 + score * 0.30), 3)
        else:               risk_level, conf = "LOW",    round(min(0.99, 0.80 + (0.20 - score)), 3)

        if not flags: flags.append("No significant risk indicators detected")

        return {
            "risk_level": risk_level, "confidence": conf, "score": round(score, 4),
            "flags": flags, "keyword_hits": {"high": high_hits, "medium": med_hits},
        }


# ---------------------------------------------------------------------------
# SemanticSearchEngine
# ---------------------------------------------------------------------------
class SemanticSearchEngine:
    """
    In-memory semantic search using sentence-transformer embeddings.
    Documents are encoded with all-MiniLM-L6-v2 and ranked by cosine similarity.
    """
    def __init__(self):
        self._docs: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        model = _get_embedding_model()
        if model is None:
            logger.warning(f"Embedding model unavailable; {doc_id} not indexed.")
            return
        snippet = text[:200].strip().replace("\n", " ") + ("…" if len(text) > 200 else "")
        vec = np.array(model.encode(text[:2000], normalize_embeddings=True), dtype=np.float32).reshape(1, -1)
        self._docs.append({"id": doc_id, "metadata": metadata or {}, "snippet": snippet})
        self._embeddings = vec if self._embeddings is None else np.vstack([self._embeddings, vec])
        logger.info(f"Indexed {doc_id}. Index size: {len(self._docs)}")

    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self._embeddings is None or not self._docs: return []
        model = _get_embedding_model()
        if model is None:
            logger.warning("Embedding model unavailable; empty search results.")
            return []
        qv    = np.array(model.encode(query, normalize_embeddings=True), dtype=np.float32)
        scores = self._embeddings @ qv
        top    = np.argsort(scores)[::-1][:min(k, len(self._docs))]
        return [{"id": self._docs[i]["id"], "score": round(float(scores[i]), 4),
                 "snippet": self._docs[i]["snippet"], "metadata": self._docs[i]["metadata"]} for i in top]

    @property
    def index_size(self) -> int:
        return len(self._docs)
