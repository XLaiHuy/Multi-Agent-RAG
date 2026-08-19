#!/usr/bin/env python3
"""
Production Admin Bootstrap Script.
Safely creates the initial admin tenant and user for production deployments (e.g. Railway / PostgreSQL).
Reads credentials strictly from environment variables without hardcoded secrets.
"""
import os
import sys
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.persistence.database import SessionLocal, UserRepository, engine
from backend.app.domain.models import Base, Tenant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bootstrap_admin")


def bootstrap_admin() -> int:
    username = os.environ.get("BOOTSTRAP_ADMIN_USERNAME", "admin").strip()
    password = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD", "").strip()
    full_name = os.environ.get("BOOTSTRAP_ADMIN_FULL_NAME", "System Administrator").strip()
    tenant_id = os.environ.get("BOOTSTRAP_ADMIN_TENANT_ID", "default_tenant").strip()

    if not username:
        logger.error("[Bootstrap] BOOTSTRAP_ADMIN_USERNAME must not be empty.")
        return 1

    if not password:
        logger.error("[Bootstrap] BOOTSTRAP_ADMIN_PASSWORD environment variable is required.")
        return 1

    if len(password) < 8:
        logger.error("[Bootstrap] BOOTSTRAP_ADMIN_PASSWORD must be at least 8 characters long.")
        return 1

    try:
        # Ensure schema tables exist
        Base.metadata.create_all(bind=engine)

        with SessionLocal() as db:
            # 1. Ensure Tenant exists
            tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
            if not tenant:
                logger.info(f"[Bootstrap] Creating tenant '{tenant_id}'...")
                tenant = Tenant(
                    id=tenant_id,
                    name=f"Enterprise Organization ({tenant_id})",
                )
                db.add(tenant)
                db.commit()

            # 2. Check if user already exists
            existing_user = UserRepository.get_by_username(db, username)
            if existing_user:
                logger.info(
                    f"[Bootstrap] User '{username}' already exists in tenant '{existing_user.tenant_id}'. "
                    "Skipping creation without modifying existing credentials."
                )
                return 0

            # 3. Create Admin User
            logger.info(f"[Bootstrap] Creating admin user '{username}' (role: admin) in tenant '{tenant_id}'...")
            user = UserRepository.create_user(
                db=db,
                username=username,
                password_plain=password,
                full_name=full_name,
                role="admin",
                tenant_id=tenant_id,
            )
            logger.info(f"[Bootstrap] Successfully created admin user '{user.username}' (ID: {user.id}).")
            return 0

    except Exception as e:
        logger.error(f"[Bootstrap] Failed to bootstrap admin: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(bootstrap_admin())
