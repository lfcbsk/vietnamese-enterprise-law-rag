from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

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
        return ChatResponse(**result)
    except Exception as error:
        error_id = str(uuid4())
        logger.exception(
            "Chat request failed [error_id=%s conversation_id=%s]",
            error_id,
            conversation_id,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Không thể xử lý câu hỏi. "
                f"Kiểm tra log backend với mã lỗi {error_id}."
            ),
        ) from error
