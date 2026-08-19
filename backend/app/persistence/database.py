"""
Database Engine & Persistence Layer for PostgreSQL / SQLite.
Includes schema management, safe session lifecycle, IDOR-protected repositories, and ACL enforcement.
"""
import os
import uuid
import datetime
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator

from sqlalchemy import create_engine, select, and_, or_
from sqlalchemy.orm import sessionmaker, Session

from backend.app.core.config import get_settings
from backend.app.core.security import hash_password
from backend.app.domain.models import (
    Base, User, Tenant, Document, DocumentVersion, DocumentACL,
    IngestionJob, Conversation, Message, AuditLog
)

logger = logging.getLogger("database")

settings = get_settings()

def normalize_database_url(url: str) -> str:
    """
    Normalizes PostgreSQL DATABASE_URL to use psycopg 3 (postgresql+psycopg://).
    Preserves SQLite, MySQL, and already-specified drivers without modification.
    """
    if not url:
        return url
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and not url.startswith("postgresql+"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


effective_database_url = normalize_database_url(settings.database_url)

# Connect args for SQLite to allow multi-threaded access in FastAPI
connect_args = {"check_same_thread": False} if effective_database_url.startswith("sqlite") else {}

# Create directory if using SQLite
if effective_database_url.startswith("sqlite"):
    db_file = effective_database_url.replace("sqlite:///", "")
    Path(db_file).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    effective_database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency: Yield a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Initializes database schema and optionally seeds dev accounts in development mode."""
    Base.metadata.create_all(bind=engine)
    
    is_prod = settings.environment.lower() in ["production", "prod"]

    with SessionLocal() as db:
        if not is_prod:
            # Check if default tenant exists
            default_tenant = db.query(Tenant).filter(Tenant.id == "default_tenant").first()
            if not default_tenant:
                default_tenant = Tenant(id="default_tenant", name="Enterprise Standard Corp")
                db.add(default_tenant)
                db.commit()

            # Check if users already seeded
            user_count = db.query(User).count()
            if user_count == 0:
                logger.info("[Database] Seeding development accounts (admin, legal01, finance01, hr01, user01)...")
                demo_users = [
                    User(
                        username="admin",
                        hashed_password=hash_password("admin123"),
                        full_name="System Administrator",
                        role="admin",
                        tenant_id="default_tenant",
                    ),
                    User(
                        username="legal01",
                        hashed_password=hash_password("legal123"),
                        full_name="Senior Legal Counsel",
                        role="legal",
                        tenant_id="default_tenant",
                    ),
                    User(
                        username="finance01",
                        hashed_password=hash_password("finance123"),
                        full_name="Finance Director",
                        role="finance",
                        tenant_id="default_tenant",
                    ),
                    User(
                        username="hr01",
                        hashed_password=hash_password("hr123"),
                        full_name="HR Manager",
                        role="hr",
                        tenant_id="default_tenant",
                    ),
                    User(
                        username="user01",
                        hashed_password=hash_password("user123"),
                        full_name="Standard Employee",
                        role="user",
                        tenant_id="default_tenant",
                    ),
                ]
                db.add_all(demo_users)
                db.commit()
                logger.info("[Database] Development seed complete.")

            # Check if contracts already seeded
            doc_count = db.query(Document).count()
            fixtures_dir = Path("tests/fixtures/cuad_small")
            if doc_count == 0 and fixtures_dir.exists():
                logger.info("[Database] Auto-seeding CUAD demo contracts for immediate testing...")
                for md_file in sorted(fixtures_dir.glob("*.md")):
                    doc_id = md_file.stem
                    title = "Cooperation and License Agreement" if "01" in doc_id else "Enterprise Cloud Services Agreement"

                    doc = Document(
                        id=doc_id,
                        tenant_id="default_tenant",
                        filename=f"{title}.md",
                        original_filename=f"{title}.md",
                        file_type="markdown",
                        storage_path=str(md_file.resolve()),
                        char_count=len(md_file.read_text(encoding="utf-8")),
                        page_count=2,
                        created_by="admin",
                    )
                    db.add(doc)

                    for r in ["admin", "legal", "finance", "hr", "user", "*"]:
                        db.add(DocumentACL(document_id=doc_id, tenant_id="default_tenant", role=r, allow_read=True))

                db.commit()
        else:
            logger.info("[Database] Production mode active: skipping default demo user & contract seeding.")

        # Rehydrate in-memory BM25 index from persistent ChromaDB collection or disk in both dev and prod
        try:
            from backend.app.ingestion.pipeline import get_ingestion_pipeline
            pipeline = get_ingestion_pipeline()
            if pipeline.bm25.bm25 is None or len(pipeline.bm25.chunk_ids) == 0:
                # Fast path: Rehydrate directly from persistent ChromaDB collection
                try:
                    chroma_data = pipeline.dense.collection.get(include=["documents", "metadatas"])
                    if chroma_data and chroma_data.get("ids") and len(chroma_data["ids"]) > 0:
                        logger.info(f"[Database] Rehydrating BM25 index from {len(chroma_data['ids'])} chunks in persistent ChromaDB...")
                        pipeline.bm25.build_index(
                            chunk_ids=chroma_data["ids"],
                            documents=chroma_data["documents"],
                            metadatas=chroma_data["metadatas"],
                        )
                        logger.info("[Database] In-memory BM25 index rehydrated successfully from ChromaDB.")
                    else:
                        raise ValueError("Chroma collection is empty; checking database records...")
                except Exception as chroma_err:
                    logger.debug(f"[Database] Chroma rehydration skipped: {chroma_err}")
                    all_docs = db.query(Document).all()
                    if all_docs:
                        logger.info(f"[Database] Synchronizing {len(all_docs)} documents into in-memory BM25 & ChromaDB...")
                        for doc in all_docs:
                            p = Path(doc.storage_path)
                            if p.exists():
                                job_id = f"startup_sync_{doc.id}"
                                pipeline.process_job(job_id, doc.id, p, tenant_id=doc.tenant_id)
                        logger.info("[Database] Ingestion & indexing synchronized successfully!")
        except Exception as e:
            logger.warning(f"[Database] Could not sync document index on startup: {e}")


# --- Repositories & Data Access ---

class UserRepository:
    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[User]:
        return db.query(User).filter(User.username == username, User.is_active == True).first()

    @staticmethod
    def create_user(db: Session, username: str, password_plain: str, full_name: str, role: str, tenant_id: str = "default_tenant") -> User:
        user = User(
            username=username,
            hashed_password=hash_password(password_plain),
            full_name=full_name,
            role=role,
            tenant_id=tenant_id,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


class DocumentRepository:
    @staticmethod
    def list_accessible_documents(db: Session, tenant_id: str, role: str) -> List[Document]:
        """Strict ACL Query: Admin sees all; others see only documents matching their role or tenant."""
        if role == "admin":
            return db.query(Document).filter(Document.tenant_id == tenant_id).order_by(Document.created_at.desc()).all()

        # Join with DocumentACL to verify allow_read
        stmt = (
            db.query(Document)
            .join(DocumentACL, Document.id == DocumentACL.document_id)
            .filter(
                Document.tenant_id == tenant_id,
                DocumentACL.role.in_([role, "*"]),
                DocumentACL.allow_read == True
            )
            .order_by(Document.created_at.desc())
        )
        return stmt.all()

    @staticmethod
    def get_document_if_accessible(db: Session, doc_id: str, tenant_id: str, role: str) -> Optional[Document]:
        """Verifies ACL before returning document to prevent cross-tenant/cross-role access."""
        doc = db.query(Document).filter(Document.id == doc_id, Document.tenant_id == tenant_id).first()
        if not doc:
            return None
        if role == "admin":
            return doc

        acl = db.query(DocumentACL).filter(
            DocumentACL.document_id == doc_id,
            DocumentACL.role.in_([role, "*"]),
            DocumentACL.allow_read == True
        ).first()

        return doc if acl is not None else None

    @staticmethod
    def create_document(
        db: Session,
        doc_id: str,
        tenant_id: str,
        filename: str,
        original_filename: str,
        file_type: str,
        storage_path: str,
        created_by: str,
        char_count: int = 0,
        page_count: int = 1,
        is_scanned: bool = False,
        allowed_roles: Optional[List[str]] = None,
    ) -> Document:
        doc = Document(
            id=doc_id,
            tenant_id=tenant_id,
            filename=filename,
            original_filename=original_filename,
            file_type=file_type,
            storage_path=storage_path,
            char_count=char_count,
            page_count=page_count,
            is_scanned=is_scanned,
            created_by=created_by,
        )
        db.add(doc)
        
        # Initial version
        ver = DocumentVersion(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            version_number=1,
            storage_path=storage_path,
            is_current=True,
        )
        db.add(ver)

        # Set default ACL
        roles_to_grant = allowed_roles or ["admin", "legal", "finance", "hr", "user"]
        for r in roles_to_grant:
            acl = DocumentACL(
                document_id=doc_id,
                tenant_id=tenant_id,
                role=r,
                allow_read=True,
                allow_write=(r in ["admin", "legal"]),
            )
            db.add(acl)

        db.commit()
        db.refresh(doc)
        return doc


class JobRepository:
    @staticmethod
    def create_job(db: Session, job_id: str, document_id: str, tenant_id: str) -> IngestionJob:
        job = IngestionJob(
            id=job_id,
            document_id=document_id,
            tenant_id=tenant_id,
            status="QUEUED",
            progress_pct=0,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def update_job_status(
        db: Session,
        job_id: str,
        status: str,
        progress_pct: int,
        error_message: Optional[str] = None,
        meta_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[IngestionJob]:
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if job:
            job.status = status
            job.progress_pct = progress_pct
            if error_message:
                job.error_message = error_message
            if meta_info:
                job.meta_info = meta_info
            job.updated_at = datetime.datetime.utcnow()
            db.commit()
            db.refresh(job)
        return job

    @staticmethod
    def get_job(db: Session, job_id: str, tenant_id: str) -> Optional[IngestionJob]:
        return db.query(IngestionJob).filter(IngestionJob.id == job_id, IngestionJob.tenant_id == tenant_id).first()


class ConversationRepository:
    @staticmethod
    def list_user_conversations(db: Session, username: str, tenant_id: str) -> List[Conversation]:
        """Only returns conversations belonging to the authenticated user and tenant."""
        return (
            db.query(Conversation)
            .filter(Conversation.username == username, Conversation.tenant_id == tenant_id)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    @staticmethod
    def get_conversation_messages_safe(db: Session, conv_id: str, username: str, tenant_id: str) -> Optional[List[Message]]:
        """
        ANTI-IDOR VERIFICATION:
        Ensures the requested conv_id is owned by the authenticated username & tenant.
        Returns None if unauthorized.
        """
        conv = db.query(Conversation).filter(
            Conversation.id == conv_id,
            Conversation.username == username,
            Conversation.tenant_id == tenant_id
        ).first()

        if not conv:
            return None

        return db.query(Message).filter(Message.conv_id == conv_id).order_by(Message.created_at.asc()).all()

    @staticmethod
    def save_message(
        db: Session,
        conv_id: str,
        username: str,
        tenant_id: str,
        role: str,
        content: str,
        title: Optional[str] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        retrieval_path: Optional[str] = None,
        verification_status: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> Message:
        conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
        now = datetime.datetime.utcnow()

        if not conv:
            conv_title = title or (content[:45] + ("..." if len(content) > 45 else ""))
            conv = Conversation(
                id=conv_id,
                tenant_id=tenant_id,
                username=username,
                title=conv_title,
                created_at=now,
                updated_at=now,
            )
            db.add(conv)
        else:
            # Verify ownership
            if conv.username != username or conv.tenant_id != tenant_id:
                raise PermissionError("Access denied: cannot write to a conversation owned by another user.")
            conv.updated_at = now

        msg = Message(
            conv_id=conv_id,
            role=role,
            content=content,
            citations_json=citations,
            retrieval_path=retrieval_path,
            verification_status=verification_status,
            latency_ms=latency_ms,
            created_at=now,
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg

    @staticmethod
    def delete_conversation_safe(db: Session, conv_id: str, username: str, tenant_id: str) -> bool:
        """Deletes conversation only if authenticated user owns it."""
        conv = db.query(Conversation).filter(
            Conversation.id == conv_id,
            Conversation.username == username,
            Conversation.tenant_id == tenant_id
        ).first()
        if not conv:
            return False
        db.delete(conv)
        db.commit()
        return True


class AuditRepository:
    @staticmethod
    def log_event(
        db: Session,
        tenant_id: str,
        username: str,
        action: str,
        resource_type: str,
        decision: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        audit = AuditLog(
            tenant_id=tenant_id,
            username=username,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            decision=decision,
            details_json=details,
        )
        db.add(audit)
        db.commit()
        return audit
