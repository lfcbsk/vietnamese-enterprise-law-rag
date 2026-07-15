"""
Sửa lỗi OCR dính từ (tổchức -> tổ chức, vềviệc -> về việc) bằng cách tự học
từ điển âm tiết từ chính corpus, thay vì liệt kê tay từng cặp từ.

Ý tưởng:
1. Chạy qua toàn bộ text đã clean (sau PDFLoader.clean_text hiện tại), tách
   theo khoảng trắng có sẵn -> phần lớn token là ĐÚNG (vì OCR chỉ lỗi cục bộ).
2. Token ngắn (<=7 ký tự, đúng hình thái âm tiết tiếng Việt) xuất hiện lặp
   lại nhiều lần -> coi là "âm tiết hợp lệ", đưa vào từ điển kèm tần suất.
3. Với token dài bất thường (>7 ký tự, không có trong từ điển) -> chạy DP
   (word segmentation kiểu Viterbi) để tách thành chuỗi âm tiết hợp lệ có
   tần suất cao nhất. Nếu tách được phủ kín toàn bộ token -> thay bằng bản
   có dấu cách. Nếu không tách được -> để nguyên, in ra để review thủ công.

Cách dùng:
    python segment_fix.py data/ocr/59.signed.txt data/ocr/59tiep.txt

In ra bảng "trước -> sau" cho từng token đã sửa, để bạn review trước khi
áp dụng vào pipeline thật (tích hợp vào PDFLoader.clean_text).
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

VOWELS = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ"
WORD_RE = re.compile(r"[a-zA-ZđĐ" + VOWELS + VOWELS.upper() + r"]+")

MAX_SYLLABLE_LEN = 7  # âm tiết tiếng Việt hiếm khi dài hơn 7 ký tự (nghiêng/nguyễn...)
MIN_SYLLABLE_LEN = 2  # bỏ token 1 ký tự: tiếng Việt gần như không có từ 1 chữ cái đứng riêng
MIN_FREQ_TO_TRUST = 3  # token ngắn phải xuất hiện >=3 lần mới coi là "biết chắc"

HAS_DIACRITIC = re.compile("[" + VOWELS + "đĐ" + "]")

# Một số âm tiết chức năng cực phổ biến, luôn tin dù tần suất thấp trong 1 văn bản nhỏ.
SEED_SYLLABLES = {
    "và", "là", "có", "không", "của", "về", "các", "này", "đó", "cho",
    "theo", "tại", "trong", "với", "một", "những", "được", "khi", "để",
    "đã", "từ", "như", "thì", "trên", "hoặc", "nếu", "do", "bị", "tổ",
    "chức", "việc", "đó", "đông", "phần", "cổ", "ty", "công", "doanh",
    "nghiệp", "vốn", "góp", "tài", "sản", "nghĩa", "vụ", "quyền",
}


def build_syllable_dict(texts: list[str]) -> Counter[str]:
    freq: Counter[str] = Counter()
    for text in texts:
        for tok in WORD_RE.findall(text.lower()):
            if MIN_SYLLABLE_LEN <= len(tok) <= MAX_SYLLABLE_LEN:
                freq[tok] += 1
    for s in SEED_SYLLABLES:
        freq[s] += 1000  # ưu tiên cao, không bị loại vì tần suất thấp
    return freq


def segment_token(token: str, freq: Counter[str]) -> list[str] | None:
    """DP: tìm cách tách token thành chuỗi âm tiết hợp lệ có tổng log-freq cao nhất.
    Trả về None nếu không tách được phủ kín toàn bộ token."""
    n = len(token)
    # best[i] = (score, split) tốt nhất để tách token[:i]
    best: list[tuple[float, list[str]] | None] = [None] * (n + 1)
    best[0] = (0.0, [])

    for i in range(1, n + 1):
        for j in range(max(0, i - MAX_SYLLABLE_LEN), i):
            if best[j] is None:
                continue
            piece = token[j:i]
            count = freq.get(piece)
            if count is None or count < MIN_FREQ_TO_TRUST:
                continue
            import math
            score = best[j][0] + math.log(count)
            candidate = (score, best[j][1] + [piece])
            if best[i] is None or candidate[0] > best[i][0]:
                best[i] = candidate

    return best[n][1] if best[n] is not None else None


def fix_text(text: str, freq: Counter[str]) -> tuple[str, list[tuple[str, str]]]:
    changes: list[tuple[str, str]] = []

    def repl(m: re.Match[str]) -> str:
        token = m.group(0)
        if len(token) <= MAX_SYLLABLE_LEN:
            return token  # đã là 1 âm tiết bình thường, không đụng vào
        # Chuỗi dài toàn ký tự Latin không dấu (email, domain, header rác
        # kiểu "thongtinchinhphu") không phải từ tiếng Việt bị dính - đây là
        # rác cần loại bỏ ở bước khác, KHÔNG được tách âm tiết ở đây vì sẽ
        # tạo ra chuỗi vô nghĩa.
        if len(token) > 10 and not HAS_DIACRITIC.search(token):
            return token
        low = token.lower()
        pieces = segment_token(low, freq)
        if not pieces or len(pieces) < 2:
            return token
        # An toàn: không chấp nhận mảnh 1 ký tự trong kết quả tách (dấu hiệu
        # của việc dictionary bị nhiễm từ rác 1 chữ cái).
        if any(len(p) < MIN_SYLLABLE_LEN for p in pieces):
            return token
        # khôi phục hoa/thường của chữ cái đầu
        fixed = " ".join(pieces)
        if token[0].isupper():
            fixed = fixed[0].upper() + fixed[1:]
        changes.append((token, fixed))
        return fixed

    fixed_text = WORD_RE.sub(repl, text)
    return fixed_text, changes


def main(paths: list[str]) -> None:
    texts = [Path(p).read_text(encoding="utf-8") for p in paths]
    freq = build_syllable_dict(texts)
    print(f"Từ điển âm tiết học được: {len(freq)} mục\n")

    for p, text in zip(paths, texts):
        fixed, changes = fix_text(text, freq)
        print(f"=== {p}: {len(changes)} token được tách lại ===")
        for before, after in changes[:60]:
            print(f"  {before!r:30s} -> {after!r}")
        if len(changes) > 60:
            print(f"  ... và {len(changes) - 60} token khác")
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/ocr/59.signed.txt", "data/ocr/59tiep.txt"])