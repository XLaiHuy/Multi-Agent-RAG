# Security, Tenancy & Access Control

This document outlines the security controls, authentication mechanisms, tenant isolation, and anti-IDOR protections implemented across the **Enterprise Contract Intelligence Platform**.

---

## 1. Multi-Tenant Architecture & Role-Based Access Control (RBAC)

Every request to the backend is authenticated via **OAuth2 JWT Bearer tokens** containing cryptographically signed claims:
- `sub`: Unique user ID
- `username`: User account identifier
- `tenant_id`: Enterprise organization identifier
- `role`: Access role (`admin`, `legal`, `finance`, `hr`, `user`)

### Role Permissions Matrix

| Endpoint Group | Admin | Legal | Finance | HR | User |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Document Ingestion (`/api/v1/documents/upload`)** | ✅ Full | ✅ Full | ✅ Scoped | ✅ Scoped | ❌ Denied |
| **Document Deletion (`/api/v1/documents/{id}`)** | ✅ Full | ✅ Tenant | ❌ Denied | ❌ Denied | ❌ Denied |
| **Contract QA (`/api/v1/qa/ask`)** | ✅ Full | ✅ Full | ✅ Finance Docs | ✅ HR Docs | ✅ Own Docs |
| **Contract Comparison (`/api/v1/compare/`)** | ✅ Full | ✅ Full | ✅ Finance Docs | ✅ HR Docs | ❌ Denied |
| **Risk Review (`/api/v1/risk/analyze`)** | ✅ Full | ✅ Full | ❌ Denied | ❌ Denied | ❌ Denied |

---

## 2. Anti-IDOR & Document Isolation

To prevent Insecure Direct Object References (IDOR):
1. **Ownership & Tenant Verification**: When a user requests access to `/api/v1/documents/{document_id}` or initiates QA against `selected_document_id`, the system queries the relational database with compound predicates:
   ```python
   stmt = select(Document).where(
       Document.id == document_id,
       Document.tenant_id == current_user.tenant_id,
       Document.accessible_roles.contains(current_user.role)
   )
   ```
2. **Zero Cross-Tenant Leakage**:
   - **Observed zero cross-tenant retrieval leakage across 7 security and ACL regression test suites** (`tests/security/test_security_and_acl.py`).
   - Cross-tenant document IDs return HTTP `404 Not Found` rather than `403 Forbidden` to prevent object existence enumeration.

---

## 3. Cache Isolation & Namespacing

Intermediate exact and semantic caches prevent cross-department and cross-tenant data contamination by enforcing strict namespace hashing:

$$\text{CacheKey} = \text{SHA256}(\text{tenant\_id} \parallel \text{user\_role} \parallel \text{corpus\_version} \parallel \text{query\_hash})$$

A query answered for a `legal` user will never be returned from cache to an `hr` user, ensuring department confidentiality.

---

## 4. Fail-Closed Verification

The **Answer Verifier** agent audits all synthesized responses against retrieved source context:
- If citation grounding fails, the response is flagged as ungrounded.
- If upstream LLM service experiences transient timeouts or malformed outputs, the verifier fails closed with `unknown_error` rather than silently passing unverified text.
