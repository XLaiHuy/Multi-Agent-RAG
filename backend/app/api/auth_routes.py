"""
Authentication API Endpoints.
Provides OAuth2 token issuance, password validation, and user profile inspection.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.app.persistence.database import get_db, UserRepository
from backend.app.core.security import verify_password, create_access_token, get_current_user, UserTokenData
from backend.app.domain.schemas import TokenResponse, UserProfile

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Authenticate with username and password; issues JWT access token.
    """
    user = UserRepository.get_by_username(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = {
        "sub": user.username,
        "role": user.role,
        "tenant_id": user.tenant_id,
        "full_name": user.full_name,
    }
    access_token = create_access_token(token_data)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_info={
            "username": user.username,
            "role": user.role,
            "tenant_id": user.tenant_id,
            "full_name": user.full_name,
        },
    )


@auth_router.get("/me", response_model=UserProfile)
def get_profile(current_user: UserTokenData = Depends(get_current_user)):
    """Return currently authenticated user identity and role claims."""
    return UserProfile(
        username=current_user.username,
        full_name=current_user.full_name or current_user.username,
        role=current_user.role,
        tenant_id=current_user.tenant_id,
    )
