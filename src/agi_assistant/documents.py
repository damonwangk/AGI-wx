"""文档解析与本地版本化存储。"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentRecord


class DocumentLibrary:
    def __init__(self, root: str | Path = ".data/documents") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, content: str, content_type: str = "text/markdown") -> DocumentRecord:
        doc_id = hashlib.sha256(name.encode()).hexdigest()[:16]
        existing = self.get(doc_id)
        record = DocumentRecord(
            id=doc_id,
            name=name,
            content=content,
            content_type=content_type,
            version=(existing.version + 1) if existing else 1,
        )
        (self.root / f"{doc_id}.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record

    def list(self) -> list[DocumentRecord]:
        records: list[DocumentRecord] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                records.append(DocumentRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except (ValueError, json.JSONDecodeError):
                continue
        return records

    def get(self, doc_id: str) -> DocumentRecord | None:
        path = self.root / f"{doc_id}.json"
        return DocumentRecord.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    def delete(self, doc_id: str) -> bool:
        path = self.root / f"{doc_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True


def parse_document(filename: str, content_type: str, data: bytes) -> tuple[str, int, bool]:
    """解析文本或 PDF，返回正文、页数、是否需要 OCR。"""
    if filename.lower().endswith(".pdf") or content_type == "application/pdf":
        reader = PdfReader(BytesIO(data))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        return text, len(reader.pages), len(text) < max(80, len(reader.pages) * 20)
    return data.decode("utf-8", errors="replace"), 1, False
