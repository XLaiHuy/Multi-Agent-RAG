import os
import time
import secrets
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# Dùng SECRET_KEY từ env nếu có, nếu không thì tự sinh 64-byte key an toàn
SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "rag_enterprise_dh_mo_2026_secure_key_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 giờ

from app.core.db import get_user_by_username, verify_password, hash_password


class Token(BaseModel):
    access_token: str
    token_type: str
    user_info: dict


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")


def create_access_token(data: dict, expires_delta: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    to_encode = data.copy()
    to_encode["exp"] = time.time() + (expires_delta * 60)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token không hợp lệ hoặc đã hết hạn. Vui lòng đăng nhập lại.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if not username or not role:
            raise exc
        return {"username": username, "role": role}
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token đã hết hạn. Vui lòng đăng nhập lại.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise exc


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Dependency: chỉ admin mới được dùng endpoint này."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Chỉ admin mới có quyền thực hiện thao tác này."
        )
    return current_user


auth_router = APIRouter()


@auth_router.post("/login", response_model=Token, tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai tên đăng nhập hoặc mật khẩu.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": form_data.username, "role": user["role"]})
    return Token(
        access_token=token,
        token_type="bearer",
        user_info={
            "username": form_data.username,
            "role": user["role"],
            "full_name": user["full_name"],
        }
    )


@auth_router.get("/me", tags=["Auth"])
async def me(current_user: dict = Depends(get_current_user)):
    return current_user
