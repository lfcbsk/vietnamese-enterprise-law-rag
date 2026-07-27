from __future__ import annotations

import logging
import re
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


def _safe_error_reason(error: Exception) -> str | None:
    if type(error).__name__ != "ChatGoogleGenerativeAIError":
        return None

    reason = " ".join(str(error).split())
    reason = re.sub(r"AIza[\w-]+", "[API key đã ẩn]", reason)
    return reason[:500]


def _error_status_code(error: Exception) -> int:
    message = str(error).upper()
    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        return 429
    if type(error).__name__ in {"ConnectError", "TimeoutException"}:
        return 503
    return 500

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
    initialization_status = getattr(
        request.app.state,
        "initialization_status",
        "initializing",
    )
    engine = getattr(request.app.state, "rag_engine", None)
    if initialization_status != "ready" or engine is None:
        detail = (
            "Khởi tạo RAG engine thất bại. Vui lòng kiểm tra log máy chủ."
            if initialization_status == "failed"
            else "Mô hình đang khởi tạo. Vui lòng thử lại sau."
        )
        raise HTTPException(status_code=503, detail=detail)

    conversation_id = (
        payload.conversation_id
        or str(uuid4())
    )

    try:
        result = engine.ask(
            conversation_id=conversation_id,
            question=payload.question.strip(),
        )
        return ChatResponse(**result)
    except Exception as error:
        error_id = str(uuid4())
        error_type = type(error).__name__
        safe_reason = _safe_error_reason(error)
        status_code = _error_status_code(error)
        logger.exception(
            "Chat request failed "
            "[error_id=%s error_type=%s conversation_id=%s]",
            error_id,
            error_type,
            conversation_id,
        )
        raise HTTPException(
            status_code=status_code,
            detail=(
                "Không thể xử lý câu hỏi. "
                f"Loại lỗi: {error_type}. "
                f"{f'Nguyên nhân: {safe_reason}. ' if safe_reason else ''}"
                f"Mã lỗi: {error_id}."
            ),
        ) from error
