"""Unit tests for ml_pipeline/models.py — all transformer calls mocked via conftest.py"""
import sys, os, pytest
import numpy as np
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml_pipeline.models import ComplianceNERModel, RiskClassifier, SemanticSearchEngine, HIGH_RISK_KEYWORDS

SAMPLE = ("This contract between Acme Corp and Beta Ltd is subject to GDPR. "
          "A breach may result in a penalty and lawsuit. SOX audit found a violation.")
CLEAN  = "The weather today is sunny and warm."

def _mock_enc(text, normalize_embeddings=False):
    t = text if isinstance(text, str) else str(text)
    np.random.seed(abs(hash(t)) % (2**31))
    v = np.random.randn(384).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v

class TestComplianceNERModel:
    def test_empty_text_safe_defaults(self):
        m = ComplianceNERModel()
        r = m.extract_entities("")
        assert r["transformer_entities"] == [] and r["doc_type"] == "unknown"

    def test_regulation_patterns_without_transformer(self):
        with patch("ml_pipeline.models._get_ner_pipeline", return_value=None):
            r = ComplianceNERModel().extract_entities(SAMPLE)
        regs = [e["word"] for e in r["transformer_entities"] if e["entity"]=="REGULATION"]
        assert "GDPR" in regs and "SOX" in regs

    def test_doc_type_contract(self):
        with patch("ml_pipeline.models._get_ner_pipeline", return_value=None):
            r = ComplianceNERModel().extract_entities("This agreement between parties hereinafter…")
        assert r["doc_type"] == "contract"

    def test_doc_type_financial(self):
        with patch("ml_pipeline.models._get_ner_pipeline", return_value=None):
            r = ComplianceNERModel().extract_entities("Invoice #1. Amount due $5000.")
        assert r["doc_type"] == "financial"

    def test_entity_count_matches_list(self):
        with patch("ml_pipeline.models._get_ner_pipeline", return_value=None):
            r = ComplianceNERModel().extract_entities(SAMPLE)
        assert r["entity_count"] == len(r["transformer_entities"])

    def test_no_duplicate_regulations(self):
        with patch("ml_pipeline.models._get_ner_pipeline", return_value=None):
            r = ComplianceNERModel().extract_entities("GDPR GDPR GDPR applies here.")
        regs = [e["word"] for e in r["transformer_entities"] if e["entity"]=="REGULATION"]
        assert len(regs) == len(set(regs))

    def test_transformer_entities_merged(self):
        fake_ner = MagicMock(return_value=[{"entity_group":"ORG","word":"Acme Corp","score":0.98}])
        with patch("ml_pipeline.models._get_ner_pipeline", return_value=fake_ner):
            r = ComplianceNERModel().extract_entities(SAMPLE)
        types = {e["entity"] for e in r["transformer_entities"]}
        assert "ORG" in types and "REGULATION" in types


class TestRiskClassifier:
    def test_high_risk_keywords_elevate_level(self):
        clf = RiskClassifier()
        r = clf.classify_risk("breach violation penalty lawsuit fraud", [])
        assert r["risk_level"] in ("HIGH","MEDIUM")

    def test_clean_text_is_low_risk(self):
        assert RiskClassifier().classify_risk(CLEAN, [])["risk_level"] == "LOW"

    def test_multiple_regulations_increase_score(self):
        clf = RiskClassifier()
        regs = [{"entity":"REGULATION","word":w,"score":1.0,"source":"pattern"} for w in ["GDPR","SOX","HIPAA"]]
        assert clf.classify_risk(CLEAN, regs)["score"] > clf.classify_risk(CLEAN, [])["score"]

    def test_score_capped_at_one(self):
        clf = RiskClassifier()
        text = " ".join(HIGH_RISK_KEYWORDS * 5)
        assert clf.classify_risk(text, [], "audit_report")["score"] <= 1.0

    def test_confidence_in_range(self):
        for text in [SAMPLE, CLEAN, ""]:
            r = RiskClassifier().classify_risk(text, [])
            assert 0.0 <= r["confidence"] <= 1.0

    def test_is_deterministic(self):
        clf = RiskClassifier()
        r1 = clf.classify_risk(SAMPLE, [])
        r2 = clf.classify_risk(SAMPLE, [])
        assert r1["risk_level"] == r2["risk_level"] and r1["score"] == r2["score"]

    def test_flags_always_present(self):
        assert len(RiskClassifier().classify_risk(CLEAN, [])["flags"]) >= 1

    def test_empty_text_returns_low(self):
        assert RiskClassifier().classify_risk("", [])["risk_level"] == "LOW"


class TestSemanticSearchEngine:
    def _engine(self):
        mock = MagicMock(); mock.encode = _mock_enc
        e = SemanticSearchEngine()
        with patch("ml_pipeline.models._get_embedding_model", return_value=mock):
            e.add_document("d1","GDPR breach penalty contract",{"type":"contract"})
            e.add_document("d2","Weather is sunny today",{"type":"general"})
        e._mock = mock; return e

    def test_empty_index_returns_empty(self):
        assert SemanticSearchEngine().search_similar("anything") == []

    def test_index_size_increments(self):
        e = self._engine(); assert e.index_size == 2

    def test_search_count_respects_k(self):
        e = self._engine()
        with patch("ml_pipeline.models._get_embedding_model", return_value=e._mock):
            assert len(e.search_similar("compliance", k=1)) == 1

    def test_result_fields(self):
        e = self._engine()
        with patch("ml_pipeline.models._get_embedding_model", return_value=e._mock):
            for r in e.search_similar("compliance", k=2):
                assert all(f in r for f in ("id","score","snippet","metadata"))

    def test_scores_range(self):
        e = self._engine()
        with patch("ml_pipeline.models._get_embedding_model", return_value=e._mock):
            for r in e.search_similar("breach",k=2):
                assert -1.0 <= r["score"] <= 1.0

    def test_unavailable_model_returns_empty(self):
        e = SemanticSearchEngine()
        e._docs = [{"id":"x","snippet":"s","metadata":{}}]
        e._embeddings = np.zeros((1,384), dtype=np.float32)
        with patch("ml_pipeline.models._get_embedding_model", return_value=None):
            assert e.search_similar("q") == []
