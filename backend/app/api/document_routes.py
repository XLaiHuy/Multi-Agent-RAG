"""
Document Management & Upload API Endpoints.
Provides asynchronous document ingestion, progress tracking, and strict ACL-scoped document libraries.
"""
import os
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.security import get_current_user, UserTokenData, require_roles
from backend.app.persistence.database import get_db, DocumentRepository, JobRepository
from backend.app.ingestion.pipeline import get_ingestion_pipeline
from backend.app.domain.schemas import DocumentUploadResponse, IngestionJobStatus, DocumentInfo

document_router = APIRouter(prefix="/documents", tags=["Documents"])

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".json", ".docx", ".doc",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp"
}


def _run_background_ingestion(job_id: str, document_id: str, file_path: str, tenant_id: str):
    """Background task executing multi-stage document ingestion."""
    pipeline = get_ingestion_pipeline()
    pipeline.process_job(job_id, document_id, Path(file_path), tenant_id=tenant_id)


@document_router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    allowed_roles: Optional[str] = Form(default="admin,legal,finance,hr,user"),
    current_user: UserTokenData = Depends(require_roles("admin", "legal", "finance", "hr")),
    db: Session = Depends(get_db),
):
    """
    Asynchronous Document Upload:
    1. Validates extension and saves raw file to storage.
    2. Creates Document, Version, and IngestionJob database records.
    3. Dispatches background worker pipeline and returns immediately in < 50ms.
    """
    settings = get_settings()
    filename = file.filename or "unnamed_contract"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    storage_dir = Path(settings.storage_dir) / current_user.tenant_id
    storage_dir.mkdir(parents=True, exist_ok=True)

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    saved_path = storage_dir / f"{doc_id}_{filename}"

    # Save uploaded file
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    roles_list = [r.strip() for r in allowed_roles.split(",") if r.strip()]

    # Create Database Records
    doc = DocumentRepository.create_document(
        db=db,
        doc_id=doc_id,
        tenant_id=current_user.tenant_id,
        filename=filename,
        original_filename=filename,
        file_type=ext.replace(".", ""),
        storage_path=str(saved_path),
        created_by=current_user.username,
        allowed_roles=roles_list,
    )

    JobRepository.create_job(db=db, job_id=job_id, document_id=doc_id, tenant_id=current_user.tenant_id)

    # Dispatch async ingestion worker
    background_tasks.add_task(
        _run_background_ingestion,
        job_id=job_id,
        document_id=doc_id,
        file_path=str(saved_path),
        tenant_id=current_user.tenant_id,
    )

    return DocumentUploadResponse(
        status="success",
        document_id=doc_id,
        job_id=job_id,
        filename=filename,
        message="Document uploaded successfully. Ingestion job queued in background.",
    )


@document_router.get("/jobs/{job_id}", response_model=IngestionJobStatus)
def get_job_status(
    job_id: str,
    current_user: UserTokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Poll ingestion job progress and status."""
    job = JobRepository.get_job(db, job_id, current_user.tenant_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found.")

    return IngestionJobStatus(
        job_id=job.id,
        document_id=job.document_id,
        status=job.status,
        progress_pct=job.progress_pct,
        error_message=job.error_message,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
    )


@document_router.get("", response_model=List[DocumentInfo])
def list_documents(
    current_user: UserTokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all documents accessible to the authenticated user based on role and tenant ACL.
    """
    docs = DocumentRepository.list_accessible_documents(db, current_user.tenant_id, current_user.role)
    result = []
    for d in docs:
        result.append(
            DocumentInfo(
                id=d.id,
                filename=d.filename,
                file_type=d.file_type,
                char_count=d.char_count,
                page_count=d.page_count,
                is_scanned=d.is_scanned,
                created_by=d.created_by,
                created_at=d.created_at.isoformat(),
                status="READY",
            )
        )
    return result


@document_router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(
    doc_id: str,
    current_user: UserTokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve document metadata with strict ACL verification."""
    doc = DocumentRepository.get_document_if_accessible(db, doc_id, current_user.tenant_id, current_user.role)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Document not found or you lack authorized access permissions.",
        )

    return DocumentInfo(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        char_count=doc.char_count,
        page_count=doc.page_count,
        is_scanned=doc.is_scanned,
        created_by=doc.created_by,
        created_at=doc.created_at.isoformat(),
        status="READY",
    )
