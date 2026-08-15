"""
Canonical Document Model.
Provides format-independent representation of contracts and multi-modal documents.
Normalizes PDF, scanned PDF, Markdown, JSON, DOCX, and images into structured hierarchy:
CanonicalDocument -> CanonicalPage -> CanonicalBlock (with bboxes, offsets, section paths).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CLAUSE = "clause"
    TABLE = "table"
    LIST_ITEM = "list_item"
    CODE = "code"
    IMAGE_DESCRIPTION = "image_description"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box coordinates on a page (x0, y0, x1, y1) in points or normalized [0, 1]."""
    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> Dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        return cls(
            x0=float(data.get("x0", 0.0)),
            y0=float(data.get("y0", 0.0)),
            x1=float(data.get("x1", 0.0)),
            y1=float(data.get("y1", 0.0)),
        )


@dataclass
class CanonicalBlock:
    """Atomic content block within a page/section."""
    block_id: str
    block_type: BlockType
    text: str
    page_number: int
    section_path: List[str] = field(default_factory=list) # e.g. ["Article 8", "8.2 Termination"]
    source_offset_start: int = 0
    source_offset_end: int = 0
    bbox: Optional[BoundingBox] = None
    table_data: Optional[List[List[str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "block_type": self.block_type.value,
            "text": self.text,
            "page_number": self.page_number,
            "section_path": self.section_path,
            "source_offset_start": self.source_offset_start,
            "source_offset_end": self.source_offset_end,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "table_data": self.table_data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalBlock":
        bbox_data = data.get("bbox")
        return cls(
            block_id=data["block_id"],
            block_type=BlockType(data.get("block_type", "paragraph")),
            text=data.get("text", ""),
            page_number=int(data.get("page_number", 1)),
            section_path=data.get("section_path", []),
            source_offset_start=int(data.get("source_offset_start", 0)),
            source_offset_end=int(data.get("source_offset_end", 0)),
            bbox=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            table_data=data.get("table_data"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CanonicalPage:
    """Single page or logical sheet of a document."""
    page_number: int
    width: float = 612.0
    height: float = 792.0
    blocks: List[CanonicalBlock] = field(default_factory=list)
    is_scanned: bool = False
    ocr_confidence: Optional[float] = None
    image_uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "blocks": [b.to_dict() for b in self.blocks],
            "is_scanned": self.is_scanned,
            "ocr_confidence": self.ocr_confidence,
            "image_uri": self.image_uri,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalPage":
        return cls(
            page_number=int(data["page_number"]),
            width=float(data.get("width", 612.0)),
            height=float(data.get("height", 792.0)),
            blocks=[CanonicalBlock.from_dict(b) for b in data.get("blocks", [])],
            is_scanned=bool(data.get("is_scanned", False)),
            ocr_confidence=data.get("ocr_confidence"),
            image_uri=data.get("image_uri"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CanonicalDocument:
    """Top-level canonical document representation."""
    doc_id: str
    title: str
    doc_type: str # pdf | markdown | json | docx | image
    pages: List[CanonicalPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_all_blocks(self) -> List[CanonicalBlock]:
        all_blocks = []
        for page in self.pages:
            all_blocks.extend(page.blocks)
        return all_blocks

    def get_full_text(self) -> str:
        page_texts = [p.get_full_text() for p in self.pages]
        return "\n\n".join(t for t in page_texts if t.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "pages": [p.to_dict() for p in self.pages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalDocument":
        return cls(
            doc_id=data["doc_id"],
            title=data.get("title", ""),
            doc_type=data.get("doc_type", "unknown"),
            pages=[CanonicalPage.from_dict(p) for p in data.get("pages", [])],
            metadata=data.get("metadata", {}),
        )
