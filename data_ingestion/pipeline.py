"""
Document ingestion pipeline — handles PDFs, images, and text files.
Optional dependencies (pdfplumber, fitz, pytesseract) degrade gracefully.
"""
import os, io, logging, hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import pdfplumber; _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    import fitz; _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

try:
    import pytesseract
    from PIL import Image
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False


@dataclass
class RawDocument:
    id: str; content: bytes; filename: str
    file_type: str; metadata: Dict[str, Any]; upload_timestamp: datetime


class DocumentIngestor:
    """Ingests PDFs, images, and text files with graceful OCR fallback."""

    def __init__(self, config: Dict):
        self.config = config

    def ingest_document(self, file_path: str) -> RawDocument:
        with open(file_path, "rb") as f:
            content = f.read()
        doc_id   = f"doc_{hashlib.sha256(content).hexdigest()[:16]}"
        metadata = self._extract_metadata(file_path, content)
        return RawDocument(
            id=doc_id, content=self._process_content(content, metadata["file_type"]),
            filename=os.path.basename(file_path), file_type=metadata["file_type"],
            metadata=metadata, upload_timestamp=datetime.utcnow(),
        )

    def _extract_metadata(self, file_path: str, content: bytes) -> Dict[str, Any]:
        ext = os.path.splitext(file_path)[-1].lstrip(".").lower()
        ft  = "image" if ext in ("jpg","jpeg","png","gif","bmp","tiff") else \
              "pdf"   if ext == "pdf" else \
              "text"  if ext in ("txt","md","csv") else "unknown"
        try:   mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        except: mtime = datetime.utcnow().isoformat()
        return {"size_bytes": len(content), "file_type": ft, "extension": ext,
                "last_modified": mtime, "quality_score": 0.85, "is_scanned": ft == "image"}

    def _process_content(self, content: bytes, file_type: str) -> bytes:
        return {"pdf": self._process_pdf, "image": self._process_image,
                "text": lambda c: c.decode("utf-8", errors="replace").encode()
                }.get(file_type, lambda c: c)(content)

    def _process_pdf(self, content: bytes) -> bytes:
        if _HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    parts = [p.extract_text() for p in pdf.pages if p.extract_text()]
                    if parts: return "\n\n".join(parts).encode()
            except Exception: pass
        if _HAS_FITZ and _HAS_OCR:
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                parts = []
                for n in range(len(doc)):
                    pix = doc[n].get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    parts.append(pytesseract.image_to_string(img))
                if parts: return "\n\n".join(parts).encode()
            except Exception: pass
        return content

    def _process_image(self, content: bytes) -> bytes:
        if not _HAS_OCR: return b""
        try:
            return pytesseract.image_to_string(Image.open(io.BytesIO(content))).encode()
        except Exception: return b""
