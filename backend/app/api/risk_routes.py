"""
Contract Risk Review API Endpoints.
Performs deterministic rule checks combined with LLM contextual risk auditing.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.security import get_current_user, UserTokenData
from backend.app.persistence.database import get_db, DocumentRepository
from backend.app.application.contract_risk import get_contract_risk_service
from backend.app.domain.schemas import RiskReviewRequest, RiskReviewResponse

risk_router = APIRouter(prefix="/risk", tags=["Contract Risk Review"])


@risk_router.post("/review", response_model=RiskReviewResponse)
def review_contract_risks_endpoint(
    request_data: RiskReviewRequest,
    current_user: UserTokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Executes hybrid rule-based and LLM risk assessment for a contract.
    """
    doc = DocumentRepository.get_document_if_accessible(
        db, request_data.document_id, current_user.tenant_id, current_user.role
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied or Document ({request_data.document_id}) not found.",
        )

    risk_service = get_contract_risk_service()

    result = risk_service.review_contract_risks(
        document_id=request_data.document_id,
        document_name=doc.filename,
        tenant_id=current_user.tenant_id,
        role=current_user.role,
        username=current_user.username,
        custom_rule_ids=request_data.custom_rules,
    )

    return result
