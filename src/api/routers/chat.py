from __future__ import annotations

from uuid import uuid4

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)

from src.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    conversation_id = (
        payload.conversation_id
        or str(uuid4())
    )

    try:
        result = request.app.state.rag_engine.ask(
            conversation_id=conversation_id,
            question=payload.question.strip(),
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Không thể xử lý câu hỏi.",
        ) from error

    return ChatResponse(**result)