from fastapi import APIRouter, HTTPException
from models.request_models import DiagramGenerationRequest
from models.response_models import DiagramGenerationResponse
from services.gemini_service import generate_diagram

router = APIRouter(prefix="/api/ia", tags=["diagram"])


@router.post("/generate-diagram", response_model=DiagramGenerationResponse)
async def generate_diagram_endpoint(
    request: DiagramGenerationRequest
):
    try:
        result = await generate_diagram(request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando diagrama: {str(e)}"
        )


@router.get("/health")
async def health():
    return {"status": "ok", "service": "ibpms-ia"}
