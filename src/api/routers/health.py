from fastapi import (
    APIRouter,
    Request,
    Response,
    status,
)


router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@router.get("")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def readiness(
    request: Request,
    response: Response,
) -> dict[str, str]:
    initialization_status = getattr(
        request.app.state,
        "initialization_status",
        "initializing",
    )
    if initialization_status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    result = {"status": initialization_status}
    initialization_error = getattr(
        request.app.state,
        "initialization_error",
        None,
    )
    if initialization_status == "failed" and initialization_error:
        result["error"] = initialization_error
    return result
