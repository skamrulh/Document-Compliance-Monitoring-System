"""
Stub out heavy optional dependencies so tests run in CI without GPU/weights.
This file is loaded by pytest automatically before any test module.
"""
import sys
from unittest.mock import MagicMock

# Stub transformers
transformers_mock = MagicMock()
transformers_mock.pipeline = MagicMock()
sys.modules.setdefault("transformers", transformers_mock)

# Stub sentence_transformers
st_mock = MagicMock()
sys.modules.setdefault("sentence_transformers", st_mock)
sys.modules.setdefault("sentence_transformers.SentenceTransformer", st_mock)

# Stub fitz (PyMuPDF)
sys.modules.setdefault("fitz", MagicMock())

# Stub pdfplumber
sys.modules.setdefault("pdfplumber", MagicMock())

# Stub pytesseract + PIL
sys.modules.setdefault("pytesseract", MagicMock())
sys.modules.setdefault("PIL", MagicMock())
sys.modules.setdefault("PIL.Image", MagicMock())
