"""
Contract Comparison API Endpoints.
Provides decomposed, facet-by-facet contract comparison and difference analysis.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.security import get_current_user, UserTokenData
from backend.app.persistence.database import get_db, DocumentRepository
from backend.app.application.contract_compare import get_contract_compare_service
from backend.app.domain.schemas import ContractCompareRequest, ContractCompareResponse

compare_router = APIRouter(prefix="/compare", tags=["Contract Comparison"])


@compare_router.post("", response_model=ContractCompareResponse)
def compare_contracts_endpoint(
    request_data: ContractCompareRequest,
    current_user: UserTokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Compares two contracts across standard or customized legal facets.
    Retrieves evidence independently per contract and synthesizes a structured contrast matrix.
    """
    # Verify access to Contract A
    doc_a = DocumentRepository.get_document_if_accessible(
        db, request_data.contract_a_id, current_user.tenant_id, current_user.role
    )
    if not doc_a:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied or Contract A ({request_data.contract_a_id}) not found.",
        )

    # Verify access to Contract B
    doc_b = DocumentRepository.get_document_if_accessible(
        db, request_data.contract_b_id, current_user.tenant_id, current_user.role
    )
    if not doc_b:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied or Contract B ({request_data.contract_b_id}) not found.",
        )

    compare_service = get_contract_compare_service()

    result = compare_service.compare_contracts(
        contract_a_id=request_data.contract_a_id,
        contract_b_id=request_data.contract_b_id,
        contract_a_name=doc_a.filename,
        contract_b_name=doc_b.filename,
        tenant_id=current_user.tenant_id,
        role=current_user.role,
        username=current_user.username,
        custom_facets=request_data.facets,
    )

    return result
