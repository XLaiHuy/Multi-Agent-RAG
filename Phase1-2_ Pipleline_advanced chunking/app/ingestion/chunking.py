from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_document(
    doc_id: str,
    text: str,
    source: str,
    extra_metadata: dict | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 75,
) -> list[Chunk]:
    if not text.strip():
        raise ValueError(f"Cannot chunk empty text for doc_id: {doc_id}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pieces = splitter.split_text(text)
    if not pieces:
        raise ValueError(f"Splitter generated 0 chunks for document '{doc_id}'")

    chunks = []
    base_meta = extra_metadata.copy() if extra_metadata else {}

    for i, piece in enumerate(pieces):
        chunk_id = f"{doc_id}_chunk_{i}"
        chunk_meta = {
            **base_meta,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source": source,
            "filename": base_meta.get("filename", source),
            "chunk_index": i,
        }

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=piece,
                source=source,
                chunk_index=i,
                metadata=chunk_meta,
            )
        )

    return chunks
