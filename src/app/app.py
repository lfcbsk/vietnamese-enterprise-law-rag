from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

ChatHandler = Callable[[str, str | None], dict[str, Any]]
ReadinessHandler = Callable[[], tuple[bool, str]]

EXAMPLES = [
    "Điều 111 quy định gì?",
    "Khoản 2 Điều 17 nói gì?",
    "Quyền của cổ đông phổ thông là gì?",
    "So sánh Điều 111 với Điều 120.",
]

CSS = """
.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
}
.legal-note {
    border-left: 4px solid var(--primary-500);
    padding: 0.65rem 0.9rem;
    background: var(--background-fill-secondary);
    border-radius: 0.4rem;
}
"""


def _format_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""

    lines = ["", "---", "**Nguồn tham khảo**"]
    for source in sources:
        article = source.get("article") or "Không rõ điều"
        title = source.get("article_title")
        pages = _format_pages(source.get("page_start"), source.get("page_end"))

        description = str(article)
        if title:
            description += f" — {title}"
        if pages:
            description += f" ({pages})"
        lines.append(f"- {description}")

    return "\n".join(lines)


def _format_pages(page_start: Any, page_end: Any) -> str:
    if not page_start:
        return ""
    if page_end and page_end != page_start:
        return f"trang {page_start}–{page_end}"
    return f"trang {page_start}"


def handle_message(
    message: str,
    history: list[dict[str, Any]] | None,
    conversation_id: str | None,
    chat_handler: ChatHandler,
) -> tuple[str, list[dict[str, Any]], str | None, str]:
    """Handle one Gradio turn while keeping the callback easy to unit test."""
    question = message.strip()
    current_history = list(history or [])
    if not question:
        return (
            "",
            current_history,
            conversation_id,
            "⚠️ Vui lòng nhập câu hỏi.",
        )

    current_history.append({"role": "user", "content": question})

    try:
        result = chat_handler(question, conversation_id)
    except Exception:
        logger.exception("Gradio chat request failed")
        current_history.append(
            {
                "role": "assistant",
                "content": (
                    "Không thể xử lý câu hỏi lúc này. Mô hình có thể đang "
                    "khởi tạo hoặc dịch vụ LLM đang tạm thời không khả dụng. "
                    "Vui lòng thử lại sau."
                ),
            }
        )
        return (
            "",
            current_history,
            conversation_id,
            "❌ Yêu cầu thất bại. Xem log máy chủ để biết chi tiết.",
        )

    answer = str(result.get("answer") or "Không nhận được câu trả lời.")
    sources = result.get("sources") or []
    current_history.append(
        {
            "role": "assistant",
            "content": answer + _format_sources(sources),
        }
    )
    next_conversation_id = result.get("conversation_id") or conversation_id

    return (
        "",
        current_history,
        next_conversation_id,
        f"✅ Đã trả lời · Phiên: `{next_conversation_id}`",
    )


def create_demo(
    chat_handler: ChatHandler,
    readiness_handler: ReadinessHandler,
) -> gr.Blocks:
    """Create a Gradio UI that is mounted by the FastAPI application."""

    def submit_message(
        message: str,
        history: list[dict[str, Any]] | None,
        conversation_id: str | None,
    ) -> tuple[str, list[dict[str, Any]], str | None, str]:
        return handle_message(
            message,
            history,
            conversation_id,
            chat_handler,
        )

    def reset_conversation() -> tuple[list[Any], None, str, str]:
        return [], None, "", "🆕 Đã tạo phiên hội thoại mới."

    def show_readiness() -> str:
        ready, detail = readiness_handler()
        icon = "✅" if ready else "⏳"
        return f"{icon} {detail}"

    with gr.Blocks(title="Trợ lý Luật Doanh nghiệp Việt Nam") as demo:
        gr.Markdown(
            """
            # ⚖️ Trợ lý Luật Doanh nghiệp Việt Nam

            Tra cứu Luật Doanh nghiệp 2020 bằng RAG, có dẫn nguồn theo điều luật.
            """
        )
        gr.Markdown(
            (
                "Câu trả lời chỉ mang tính tham khảo, không thay thế tư vấn "
                "pháp lý từ người có chuyên môn."
            ),
            elem_classes=["legal-note"],
        )

        conversation_id = gr.State(value=None)
        status = gr.Markdown("⏳ Đang kiểm tra trạng thái mô hình...")
        chatbot = gr.Chatbot(
            height=520,
            layout="bubble",
            placeholder="Hãy chọn một câu hỏi mẫu hoặc nhập câu hỏi pháp lý.",
            buttons=["copy"],
        )

        with gr.Row():
            question = gr.Textbox(
                placeholder="Nhập câu hỏi pháp lý...",
                label="Câu hỏi",
                lines=2,
                max_lines=5,
                scale=8,
            )
            submit = gr.Button("Gửi", variant="primary", scale=1)

        with gr.Row():
            new_conversation = gr.Button("Tạo hội thoại mới")
            gr.Markdown("[API docs](/docs) · [Health](/health)")

        gr.Examples(
            examples=EXAMPLES,
            inputs=question,
            label="Câu hỏi gợi ý",
        )

        outputs = [question, chatbot, conversation_id, status]
        submit.click(
            submit_message,
            inputs=[question, chatbot, conversation_id],
            outputs=outputs,
        )
        question.submit(
            submit_message,
            inputs=[question, chatbot, conversation_id],
            outputs=outputs,
        )
        new_conversation.click(
            reset_conversation,
            outputs=[chatbot, conversation_id, question, status],
        )
        demo.load(show_readiness, outputs=status)

    return demo.queue(default_concurrency_limit=2)
