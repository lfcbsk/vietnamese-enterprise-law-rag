from __future__ import annotations

import re
import unicodedata

# Các từ dừng phổ biến trong văn bản pháp luật tiếng Việt. Không loại bỏ
# quá nhiều vì một số từ như "không", "có", "và", "của" vẫn giúp phân biệt
# ý nghĩa trong câu hỏi ngắn (VD "được" vs "không được").
STOPWORDS: frozenset[str] = frozenset(
    {
        "là", "và", "của", "các", "này", "đó", "cho", "theo", "tại",
        "trong", "với", "một", "những", "được", "khi", "để", "đã",
        "về", "từ", "như", "thì", "trên", "hoặc", "nếu",
    }
)

_TOKEN_PATTERN = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def _strip_accents(token: str) -> str:
    normalized = unicodedata.normalize("NFD", token)

    return "".join(
        char
        for char in normalized
        if unicodedata.category(char) != "Mn"
    )


def vietnamese_tokenize(
    text: str,
    *,
    remove_stopwords: bool = False,
    fold_accents: bool = False,
) -> list[str]:
    """Tokenize văn bản tiếng Việt cho BM25.

    MVP dùng tách từ đơn giản theo regex (mỗi âm tiết là một token), vì
    tiếng Việt phân tách âm tiết bằng khoảng trắng. Cách này không nhận ra
    từ ghép nhiều âm tiết ("doanh nghiệp" thành 2 token riêng: "doanh",
    "nghiệp"), nhưng BM25 vẫn hoạt động tốt ở mức âm tiết cho văn bản pháp
    luật vì thuật ngữ pháp lý thường lặp lại nguyên cụm.

    Nếu cần độ chính xác cao hơn (ghép đúng "doanh nghiệp" thành 1 token),
    có thể thay bằng thư viện tách từ như `underthesea` hoặc `pyvi` sau này
    mà không phải đổi API của hàm này.
    """
    text = text.lower()
    tokens = _TOKEN_PATTERN.findall(text)

    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]

    if fold_accents:
        tokens = [_strip_accents(token) for token in tokens]

    return tokens