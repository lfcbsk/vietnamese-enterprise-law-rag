from src.ingest.segment_fix import build_syllable_dict, fix_text


def test_fixes_glued_legal_terms_without_splitting_valid_syllables() -> None:
    corpus = [
        "tổ chức về việc sử dụng cổ phần doanh nghiệp chủ doanh nghiệp " * 4
    ]
    frequency = build_syllable_dict(corpus)

    fixed, changes = fix_text(
        "tổchức vềviệc sửdụng cổphần doanhnghiệp chủdoanh",
        frequency,
    )

    assert fixed == (
        "tổ chức về việc sử dụng cổ phần doanh nghiệp chủ doanh"
    )
    assert changes


def test_does_not_split_single_vietnamese_syllables() -> None:
    corpus = ["doanh anh nghiêng quyền doanh nghiệp"] * 4
    frequency = build_syllable_dict(corpus)

    fixed, changes = fix_text("doanh nghiêng quyền", frequency)

    assert fixed == "doanh nghiêng quyền"
    assert changes == []
