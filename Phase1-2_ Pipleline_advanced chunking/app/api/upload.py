"""
Admin-only File Upload API with pdf-inspector OCR integration
"""
import os
import shutil
import asyncio
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.api.auth import require_admin

upload_router = APIRouter()

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"}


@upload_router.post("/upload", tags=["Admin"])
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_admin),
):
    """
    Admin-only: Upload tài liệu PDF/TXT/MD hoặc Hình ảnh (PNG/JPG/WEBP) và index vào ChromaDB.
    Tích hợp Multimodal Vision OCR tự động trích xuất văn bản, bảng số liệu và sơ đồ.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Chỉ hỗ trợ: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Lưu file tạm
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Chạy ingestion trong thread pool để không block event loop
    try:
        result = await asyncio.to_thread(_run_ingestion, save_path)
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "documents_loaded": result.get("documents_loaded", 0),
            "chunks_generated": result.get("chunks_generated", 0),
            "chunks_embedded": result.get("chunks_embedded", 0),
        })
    except Exception as e:
        # Xóa file nếu ingestion thất bại
        if save_path.exists():
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Lỗi ingestion: {str(e)}")


def _run_ingestion(file_path: Path) -> dict:
    """Chạy ingestion pipeline cho một file đơn lẻ."""
    from app.ingestion.pipeline import IngestionPipeline
    from app.graph.agentic_rag import get_hybrid_retriever
    pipeline = IngestionPipeline()
    res = pipeline.ingest_file(file_path)
    try:
        get_hybrid_retriever().reload_index()
    except Exception as e:
        print(f"[Upload] Warning reloading BM25 index: {e}")
    return res


@upload_router.get("/documents", tags=["Admin"])
async def list_documents(current_user: dict = Depends(require_admin)):
    """Admin-only: Liệt kê các tài liệu đã upload."""
    files = []
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append({
                    "name": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "extension": f.suffix,
                })
    return {"documents": files, "total": len(files)}
