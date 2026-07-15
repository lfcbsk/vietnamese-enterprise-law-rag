"""Sửa thận trọng các âm tiết tiếng Việt bị OCR dính liền nhau."""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

VOWELS = (
    "aàáảãạăằắẳẵặâầấẩẫậ"
    "eèéẻẽẹêềếểễệ"
    "iìíỉĩị"
    "oòóỏõọôồốổỗộơờớởỡợ"
    "uùúủũụưừứửữự"
    "yỳýỷỹỵ"
)
ACCENTED_LETTERS = VOWELS.replace("a", "").replace("e", "").replace(
    "i", ""
).replace("o", "").replace("u", "").replace("y", "")

WORD_RE = re.compile(
    r"[a-zA-ZđĐ" + VOWELS + VOWELS.upper() + r"]+"
)
ACCENTED_RE = re.compile(
    "[" + ACCENTED_LETTERS + "đĐ" + "]"
)
BASE_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")

MAX_SYLLABLE_LEN = 8
MIN_SYLLABLE_LEN = 2
MIN_FREQ_TO_TRUST = 3

# Các âm tiết pháp lý thường gặp được dùng làm prior. Chúng vẫn phải ghép
# thành phương án ít mảnh nhất, nên "doanh" không bị tách thành "do anh".
SEED_SYLLABLES = {
    "ai", "anh", "ban", "bị", "các", "cho", "chủ", "chức", "có",
    "cổ", "công", "của", "do", "doanh", "dụng", "đã", "để", "đó",
    "đông", "được", "góp", "hoặc", "khi", "không", "là", "luật",
    "một", "nghiệp", "nghĩa", "này", "nếu", "như", "những", "phần",
    "quyền", "sản", "sổ", "sử", "tài", "tại", "theo", "thì", "tổ",
    "trên", "trong", "từ", "ty", "việc", "và", "về", "vốn", "vụ",
    "với",
}


def _plain_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(
        char
        for char in normalized
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


def _vowel_group_count(token: str) -> int:
    """Đếm cụm nguyên âm; một âm tiết Việt hợp lệ thường có đúng một cụm."""
    return len(BASE_VOWEL_GROUP_RE.findall(_plain_text(token)))


def build_syllable_dict(texts: list[str]) -> Counter[str]:
    """Học âm tiết từ corpus nhưng loại token có dấu hiệu nhiều từ bị dính."""
    frequency: Counter[str] = Counter()

    for text in texts:
        for token in WORD_RE.findall(text.lower()):
            if not MIN_SYLLABLE_LEN <= len(token) <= MAX_SYLLABLE_LEN:
                continue
            if _vowel_group_count(token) != 1:
                continue
            frequency[token] += 1

    for syllable in SEED_SYLLABLES:
        frequency[syllable] += 1000

    return frequency


def segment_token(
    token: str,
    frequency: Counter[str],
) -> list[str] | None:
    """Tách token bằng DP, ưu tiên ít mảnh rồi mới xét tần suất corpus."""
    length = len(token)
    best: list[tuple[int, float, list[str]] | None] = [
        None
    ] * (length + 1)
    best[0] = (0, 0.0, [])

    for end in range(1, length + 1):
        for start in range(max(0, end - MAX_SYLLABLE_LEN), end):
            previous = best[start]
            if previous is None:
                continue

            piece = token[start:end]
            count = frequency.get(piece, 0)
            if count < MIN_FREQ_TO_TRUST:
                continue
            if _vowel_group_count(piece) != 1:
                continue

            candidate = (
                previous[0] + 1,
                previous[1] + math.log(count),
                previous[2] + [piece],
            )
            current = best[end]
            if (
                current is None
                or candidate[0] < current[0]
                or (
                    candidate[0] == current[0]
                    and candidate[1] > current[1]
                )
            ):
                best[end] = candidate

    result = best[length]
    if result is None or result[0] < 2:
        return None
    return result[2]


def fix_text(
    text: str,
    frequency: Counter[str],
) -> tuple[str, list[tuple[str, str]]]:
    """Thêm khoảng trắng cho token dính; không sửa ký tự hay dấu tiếng Việt."""
    changes: list[tuple[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)

        # Một cụm nguyên âm là dấu hiệu của một âm tiết đơn, không được tách.
        if _vowel_group_count(token) < 2:
            return token

        # Chuỗi dài hoàn toàn không dấu thường là URL/header rác. Giữ nguyên để
        # bước làm sạch chuyên biệt xử lý, tránh tạo câu giả có vẻ hợp lệ.
        if len(token) > 10 and not ACCENTED_RE.search(token):
            return token

        pieces = segment_token(token.lower(), frequency)
        if pieces is None:
            return token

        fixed = " ".join(pieces)
        if token[0].isupper():
            fixed = fixed[0].upper() + fixed[1:]

        if fixed.lower() == token.lower():
            return token

        changes.append((token, fixed))
        return fixed

    return WORD_RE.sub(replace, text), changes


def main(paths: list[str]) -> None:
    texts = [Path(path).read_text(encoding="utf-8") for path in paths]
    frequency = build_syllable_dict(texts)
    print(f"Từ điển âm tiết học được: {len(frequency)} mục\n")

    for path, text in zip(paths, texts, strict=True):
        _, changes = fix_text(text, frequency)
        print(f"=== {path}: {len(changes)} token được tách lại ===")
        for before, after in changes[:60]:
            print(f"  {before!r:30s} -> {after!r}")
        if len(changes) > 60:
            print(f"  ... và {len(changes) - 60} token khác")
        print()


if __name__ == "__main__":
    main(
        sys.argv[1:]
        or ["data/ocr/59.signed.txt", "data/ocr/59tiep.txt"]
    )
