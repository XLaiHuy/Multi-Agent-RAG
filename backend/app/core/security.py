"""
Security and Authentication Module.
Provides safe JWT creation/validation, bcrypt password hashing, and role/tenant context extraction.
"""
import time
import bcrypt
import jwt
from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from backend.app.core.config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


class UserTokenData(BaseModel):
    username: str
    role: str
    tenant_id: str = "default_tenant"
    full_name: Optional[str] = None


def hash_password(plain_password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def create_access_token(
    data: Dict[str, Any], expires_delta_minutes: Optional[int] = None
) -> str:
    """Create signed JWT access token."""
    settings = get_settings()
    to_encode = data.copy()
    expire_minutes = expires_delta_minutes or settings.access_token_expire_minutes
    expire_timestamp = time.time() + (expire_minutes * 60)
    to_encode.update({"exp": expire_timestamp, "iat": time.time()})
    return jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT access token."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(token: str = Depends(oauth2_scheme)) -> UserTokenData:
    """Dependency: Extract authenticated user context from Bearer token."""
    payload = decode_access_token(token)
    username = payload.get("sub")
    role = payload.get("role")
    tenant_id = payload.get("tenant_id", "default_tenant")
    full_name = payload.get("full_name")

    if not username or not role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing required user identity fields.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UserTokenData(
        username=username, role=role, tenant_id=tenant_id, full_name=full_name
    )


def require_roles(*allowed_roles: str):
    """Dependency factory: require current user to possess one of the allowed roles."""
    def role_checker(current_user: UserTokenData = Depends(get_current_user)) -> UserTokenData:
        if current_user.role == "admin":
            return current_user
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Requires one of roles: {', '.join(allowed_roles)} (User role: {current_user.role})",
            )
        return current_user

    return role_checker
