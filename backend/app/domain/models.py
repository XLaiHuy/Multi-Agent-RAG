"""
Database Models for PostgreSQL & SQLite persistence.
Includes Users, Tenants, Documents, Versions, Document ACLs, Ingestion Jobs, Conversations, Messages, and Audit Logs.
"""
import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, Float, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="tenant", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(128), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(64), nullable=False, default="user") # admin | legal | finance | hr | user
    tenant_id = Column(String(64), ForeignKey("tenants.id"), nullable=False, default="default_tenant")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    tenant = relationship("Tenant", back_populates="users")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, index=True)
    tenant_id = Column(String(64), ForeignKey("tenants.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(32), nullable=False) # pdf | docx | md | json | image
    storage_path = Column(String(512), nullable=False)
    char_count = Column(Integer, default=0)
    page_count = Column(Integer, default=1)
    is_scanned = Column(Boolean, default=False)
    created_by = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    tenant = relationship("Tenant", back_populates="documents")
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    acls = relationship("DocumentACL", back_populates="document", cascade="all, delete-orphan")
    jobs = relationship("IngestionJob", back_populates="document", cascade="all, delete-orphan")


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id = Column(String(64), primary_key=True, index=True)
    document_id = Column(String(64), ForeignKey("documents.id"), nullable=False, index=True)
    version_number = Column(Integer, default=1)
    storage_path = Column(String(512), nullable=False)
    canonical_json_path = Column(String(512), nullable=True)
    chunk_count = Column(Integer, default=0)
    is_current = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="versions")


class DocumentACL(Base):
    __tablename__ = "document_acls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(64), ForeignKey("documents.id"), nullable=False, index=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    role = Column(String(64), nullable=False, index=True) # admin, legal, finance, hr, user
    allow_read = Column(Boolean, default=True)
    allow_write = Column(Boolean, default=False)

    document = relationship("Document", back_populates="acls")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(String(64), primary_key=True, index=True)
    document_id = Column(String(64), ForeignKey("documents.id"), nullable=False, index=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="QUEUED", index=True) # QUEUED, PARSING, OCR, NORMALIZING, CHUNKING, EMBEDDING, INDEXING, READY, FAILED
    progress_pct = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    meta_info = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="jobs")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True, index=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    username = Column(String(128), ForeignKey("users.username"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    task_type = Column(String(32), default="qa") # qa | compare | risk
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conv_id = Column(String(64), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(32), nullable=False) # user | assistant | system
    content = Column(Text, nullable=False)
    citations_json = Column(JSON, nullable=True) # Serialized list of CitationItem
    retrieval_path = Column(String(64), nullable=True) # direct | hybrid_fast | planned_adaptive | compare | risk
    verification_status = Column(String(32), nullable=True) # grounded | partially_grounded | unsupported | skipped | unknown_error
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    username = Column(String(128), nullable=False, index=True)
    action = Column(String(64), nullable=False) # DOCUMENT_READ, QUERY_SUBMITTED, ACL_DENIED, etc.
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(128), nullable=True)
    decision = Column(String(32), nullable=False) # ALLOW | DENY
    details_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
