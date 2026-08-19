# Railway Deployment Guide

This document outlines the architecture, environment variables, volume configuration, and bootstrap procedures for deploying the Enterprise Contract Intelligence Platform on [Railway](https://railway.app).

---

## 1. System Architecture

```
Railway Project
├── 1. PostgreSQL Service
│      └── Manages users, tenants, role ACLs, documents metadata,
│          conversations, messages, and audit logs.
│
├── 2. Persistent Backend Volume (/data)
│      ├── /data/chroma        (Persistent ChromaDB vector index)
│      ├── /data/storage       (Uploaded original contract files)
│      └── /data/huggingface   (Pretrained embedding & reranker cache)
│
├── 3. Backend Service (FastAPI)
│      ├── Local CPU BGE-Small embeddings (BAAI/bge-small-en-v1.5)
│      ├── Local CPU CrossEncoder reranker (ms-marco-TinyBERT-L-2-v2)
│      ├── In-memory Okapi BM25 index (rehydrated from Chroma on restart)
│      └── Google Gemini API for generation and agentic reasoning
│
└── 4. Frontend Service (Vite / React 19 + Caddy)
       └── Static production SPA bundle served via Caddy with SPA routing
```

---

## 2. Service Configurations

### A. PostgreSQL Service
Add a managed **PostgreSQL** database service in your Railway project. Railway automatically provides the connection string variable `${{Postgres.DATABASE_URL}}`.

---

### B. Backend Service (FastAPI)

1. **Source / Dockerfile**: Root `Dockerfile`.
2. **Persistent Volume**:
   - Mount Path: `/data`
3. **Environment Variables**:

| Variable | Value / Format | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `production` | Enables production security gates and error masking |
| `DEBUG` | `false` | Disables verbose debug outputs |
| `JWT_SECRET_KEY` | `<generate-32+-char-random-key>` | Secret key for HS256 JWT signing |
| `GEMINI_API_KEY` | `<your-gemini-api-key>` | Google Gemini API access key |
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | Railway PostgreSQL connection string (auto-normalized to psycopg3) |
| `CHROMA_PATH` | `/data/chroma` | Persistent path for vector embeddings |
| `STORAGE_DIR` | `/data/storage` | Persistent directory for uploaded contract files |
| `HF_HOME` | `/data/huggingface` | Persistent HuggingFace model cache |
| `EMBEDDING_PROVIDER` | `local` | Uses local CPU embedding provider |
| `LOCAL_EMBEDDING_MODEL`| `BAAI/bge-small-en-v1.5` | Standard CPU embedding model |
| `EMBEDDING_DIMENSION` | `384` | Embedding vector dimension |
| `ENABLE_RERANKER` | `true` | Enables CrossEncoder reranking |
| `USE_DOCLING_PARSER` | `false` | Default lightweight fast parser |
| `ALLOWED_ORIGINS` | `https://${{Frontend.RAILWAY_PUBLIC_DOMAIN}}` | Comma-separated CORS allowed origins |

---

### C. Frontend Service (React 19 / Caddy)

1. **Source / Dockerfile**: `frontend/Dockerfile` (context: `frontend`).
2. **Build Argument / Environment Variable**:

| Variable | Value / Format | Description |
| :--- | :--- | :--- |
| `VITE_API_URL` | `https://${{Backend.RAILWAY_PUBLIC_DOMAIN}}/api/v1` | Public API endpoint baked into frontend build |

> [!NOTE]
> `VITE_API_URL` is a **build-time variable** in Vite. Never put server secrets (`JWT_SECRET_KEY`, `GEMINI_API_KEY`, `DATABASE_URL`) in frontend variables.

---

## 3. One-Time Initial Admin Bootstrap

In `production` mode, the backend **does not** automatically seed demo accounts.

To create the initial admin user on a fresh PostgreSQL deployment:

1. Open the **Railway CLI** or Backend service **Exec / Console tab**.
2. Run the bootstrap command with environment variables:

```bash
BOOTSTRAP_ADMIN_USERNAME="admin" \
BOOTSTRAP_ADMIN_PASSWORD="YourSecurePassword123" \
BOOTSTRAP_ADMIN_FULL_NAME="System Administrator" \
BOOTSTRAP_ADMIN_TENANT_ID="default_tenant" \
python scripts/bootstrap_admin.py
```

The script is **idempotent**: running it again will verify the existing user without modifying or overwriting existing credentials.

---

## 4. Health & Readiness Probes

- **Process Liveness**: `GET /health` → Returns HTTP 200 with app title, version, and environment.
- **Database Readiness**: `GET /ready` → Executes lightweight SQL probe (`SELECT 1`). Returns HTTP 200 `{"status": "ready", "database": "connected"}` or HTTP 503 if disconnected.
- **Frontend Health**: `GET /health` → Returns HTTP 200 `OK` via Caddy.

---

## 5. Known Operational Characteristics & Limitations

1. **In-Process Ingestion**: Document parsing, chunking, and embedding run within the FastAPI application worker rather than an external distributed Celery/Redis queue.
2. **BM25 Rehydration**: Okapi BM25 is an in-memory sparse index. On container restart or redeploy, the backend automatically re-hydrates BM25 directly from the persistent ChromaDB collection (`/data/chroma`) in milliseconds without re-embedding.
