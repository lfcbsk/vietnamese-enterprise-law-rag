from __future__ import annotations

from typing import Any

from src.app.app import _format_sources
from src.app.app import handle_message


def test_handle_message_keeps_conversation_and_formats_sources() -> None:
    calls: list[tuple[str, str | None]] = []

    def chat_handler(
        question: str,
        conversation_id: str | None,
    ) -> dict[str, Any]:
        calls.append((question, conversation_id))
        return {
            "conversation_id": "conversation-1",
            "answer": "Câu trả lời.",
            "sources": [
                {
                    "article": "Điều 17",
                    "article_title": "Quyền thành lập doanh nghiệp",
                    "page_start": 12,
                    "page_end": 13,
                }
            ],
        }

    message, history, conversation_id, status = handle_message(
        "  Điều 17 quy định gì?  ",
        [],
        None,
        chat_handler,
    )

    assert calls == [("Điều 17 quy định gì?", None)]
    assert message == ""
    assert conversation_id == "conversation-1"
    assert "Đã trả lời" in status
    assert history[0] == {
        "role": "user",
        "content": "Điều 17 quy định gì?",
    }
    assert history[1]["role"] == "assistant"
    assert "Câu trả lời." in history[1]["content"]
    assert "Điều 17 — Quyền thành lập doanh nghiệp (trang 12–13)" in (
        history[1]["content"]
    )


def test_handle_message_preserves_session_after_error() -> None:
    def failing_handler(
        question: str,
        conversation_id: str | None,
    ) -> dict[str, Any]:
        raise RuntimeError("secret backend detail")

    _, history, conversation_id, status = handle_message(
        "Câu hỏi",
        [],
        "conversation-1",
        failing_handler,
    )

    assert conversation_id == "conversation-1"
    assert "secret backend detail" not in history[-1]["content"]
    assert "Yêu cầu thất bại" in status


def test_format_sources_handles_single_page() -> None:
    rendered = _format_sources(
        [
            {
                "article": "Điều 1",
                "page_start": 2,
                "page_end": 2,
            }
        ]
    )

    assert "Điều 1 (trang 2)" in rendered
