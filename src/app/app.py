from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Vietnamese Enterprise Law Assistant",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ Trợ lý Luật Doanh nghiệp Việt Nam")
st.caption(
    "Hệ thống RAG hỗ trợ tra cứu Luật Doanh nghiệp 2020. "
    "Câu trả lời chỉ mang tính tham khảo."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None


def call_chat_api(
    question: str,
    conversation_id: str | None,
) -> dict:
    payload = {
        "question": question,
        "conversation_id": conversation_id,
    }

    response = requests.post(
        f"{API_URL}/chat",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def get_api_error_detail(error: requests.RequestException) -> str:
    response = error.response
    if response is None:
        return str(error)

    try:
        detail = response.json().get("detail")
    except (requests.JSONDecodeError, ValueError):
        detail = None

    return str(detail or error)


with st.sidebar:
    st.subheader("Phiên hội thoại")

    if st.button("Tạo cuộc hội thoại mới", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

    st.write(
        "Conversation ID:",
        st.session_state.conversation_id or "Chưa có",
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])
        if sources:
            with st.expander("Nguồn tham khảo"):
                for source in sources:
                    article = source.get("article", "Không rõ điều")
                    law_name = source.get(
                        "law_name",
                        "Luật Doanh nghiệp 2020",
                    )
                    st.markdown(f"- **{article}** — {law_name}")

question = st.chat_input("Nhập câu hỏi pháp lý...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu văn bản pháp luật..."):
            try:
                result = call_chat_api(
                    question=question,
                    conversation_id=(
                        st.session_state.conversation_id
                    ),
                )

                answer = result.get(
                    "answer",
                    "Không nhận được câu trả lời.",
                )
                sources = result.get("sources", [])

                st.session_state.conversation_id = result.get(
                    "conversation_id"
                )

                st.markdown(answer)

                if sources:
                    with st.expander("Nguồn tham khảo"):
                        for source in sources:
                            article = source.get(
                                "article",
                                "Không rõ điều",
                            )
                            law_name = source.get(
                                "law_name",
                                "Luật Doanh nghiệp 2020",
                            )
                            st.markdown(
                                f"- **{article}** — {law_name}"
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except requests.Timeout:
                error_message = (
                    "Máy chủ phản hồi quá lâu. "
                    "Vui lòng thử lại."
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            except requests.RequestException as exc:
                error_message = (
                    "Backend không thể xử lý yêu cầu: "
                    f"{get_api_error_detail(exc)}"
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
