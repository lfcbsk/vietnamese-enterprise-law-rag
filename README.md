# Vietnamese Enterprise Law RAG

Dự án xây dựng hệ thống Retrieval-Augmented Generation (RAG) cho văn bản
pháp luật doanh nghiệp Việt Nam. Ở trạng thái hiện tại, dự án đã triển khai
pipeline đọc PDF, OCR, làm sạch văn bản, tách nội dung theo Điều và xuất dữ
liệu JSON. Các phần embedding, truy xuất vector và API đang được phát triển.

## Trạng thái hiện tại

- [x] Đọc văn bản trực tiếp từ PDF.
- [x] OCR bằng Tesseract với ngôn ngữ Việt và Anh.
- [x] Lưu và sử dụng lại OCR cache.
- [x] Làm sạch một số lỗi OCR phổ biến.
- [x] Tách văn bản thành chunk theo Chương và Điều.
- [x] Phát hiện một số dẫn chiếu đến điều luật khác.
- [x] Xuất kết quả vào `data/law_chunks.json`.
- [ ] Sinh embedding và lưu vào vector database.
- [ ] Truy xuất ngữ nghĩa.
- [ ] API hỏi đáp pháp luật.
- [ ] Đánh giá chất lượng retrieval và câu trả lời.

## Cấu trúc dự án

```text
vietnamese-enterprise-law-rag/
├── api/                          # API RAG (đang phát triển)
│   ├── routers/
│   │   ├── chat.py               # Endpoint hỏi đáp (chưa triển khai)
│   │   └── health.py             # Endpoint kiểm tra dịch vụ (chưa triển khai)
│   ├── schemas/
│   │   └── chat.py               # Schema request/response (chưa triển khai)
│   ├── services/
│   │   ├── llm_client.py         # Kết nối LLM (chưa triển khai)
│   │   ├── rag_engine.py         # Điều phối pipeline RAG (chưa triển khai)
│   │   └── retriever.py          # Truy xuất vector (chưa triển khai)
│   ├── config.py                 # Cấu hình API (chưa triển khai)
│   ├── Dockerfile
│   └── main.py                   # Entry point API (chưa triển khai)
│
├── configs/
│   └── ocr.yaml                  # Thông tin cấu hình OCR tham khảo
│
├── data/
│   ├── 59.signed.pdf             # Nguồn Luật Doanh nghiệp 2020, phần 1
│   ├── 59tiep.pdf                # Nguồn Luật Doanh nghiệp 2020, phần 2
│   ├── law_chunks.json           # Các chunk do pipeline sinh ra
│   └── ocr/                      # OCR cache dạng văn bản
│       ├── 59.signed.txt
│       └── 59tiep.txt
│
├── evaluation/                   # Đánh giá RAG (đang phát triển)
│   ├── eval_answer.py
│   ├── eval_retrieval.py
│   ├── question.json
│   └── results.md
│
├── ingest/
│   ├── pdf_loader.py             # Đọc PDF, OCR, cache và làm sạch văn bản
│   ├── parser.py                 # Tách Chương, Điều và dẫn chiếu pháp luật
│   ├── run.py                    # Entry point của ingestion pipeline
│   ├── embedder.py               # Sinh embedding (chưa triển khai)
│   └── Dockerfile
│
├── .env.example                  # Biến môi trường (chưa cấu hình)
├── docker-compose.yaml           # Docker stack (chưa cấu hình)
├── LICENSE
└── README.md
```

Các file trong `data/ocr/` và `data/law_chunks.json` là dữ liệu được sinh ra
từ pipeline, không phải mã nguồn.

## Yêu cầu

- Python 3.10 trở lên.
- Tesseract OCR.
- Tesseract language data: `vie.traineddata` và `eng.traineddata`.
- Package Python `pymupdf`.

Cài package Python cần thiết:

```powershell
python -m pip install pymupdf
```

Kiểm tra Tesseract và các language model:

```powershell
tesseract --version
tesseract --list-langs
```

Kết quả `--list-langs` cần có ít nhất `vie` và `eng`.

## Chạy ingestion

### Dùng OCR cache hiện có

```powershell
python ingest/run.py
```

Pipeline sẽ:

1. Tìm tất cả file PDF trong `data/`.
2. Dùng file tương ứng trong `data/ocr/` nếu cache tồn tại.
3. Làm sạch văn bản và tách nội dung theo Điều.
4. Gộp chunk từ tất cả PDF.
5. Kiểm tra chunk rỗng và ID trùng.
6. Ghi kết quả vào `data/law_chunks.json`.

### Ép OCR lại toàn bộ PDF

```powershell
python ingest/run.py --force-ocr
```

Tùy chọn `--force-ocr` sẽ:

- Bỏ qua OCR cache hiện tại.
- Không sử dụng lớp native text của PDF.
- OCR ảnh của toàn bộ trang ở 300 DPI.
- Ghi đè cache trong `data/ocr/`.
- Tạo lại `data/law_chunks.json`.

Quá trình này có thể mất nhiều thời gian. Sau khi có cache tốt, những lần chạy
tiếp theo nên bỏ `--force-ocr`.

Xem các tùy chọn dòng lệnh:

```powershell
python ingest/run.py --help
```

## Dữ liệu đầu ra

Mỗi phần tử trong `data/law_chunks.json` đại diện cho một Điều và có dạng rút
gọn như sau:

```json
{
  "id": "59_signed_điều_18",
  "law_id": "59/2020/QH14",
  "law_name": "Luật Doanh nghiệp 2020",
  "chapter": "Chương II",
  "article": "Điều 18",
  "article_title": "Hợp đồng trước đăng ký doanh nghiệp",
  "content": "...",
  "embedding_text": "...",
  "references": [],
  "metadata": {
    "source": "59.signed.pdf",
    "page_start": 12,
    "page_end": 13
  }
}
```

`embedding_text` là nội dung đã được ghép thêm tên luật, Chương và Điều để sử
dụng cho bước sinh embedding sau này.

## Lưu ý về chất lượng OCR

Văn bản pháp luật cần độ chính xác cao. Trước khi sinh embedding, nên kiểm tra:

- Số lượng Điều và Chương nhận diện được.
- Các ký tự OCR bất thường như `º`, `¦`, `Î`.
- Từ bị dính, mất dấu hoặc sai thứ tự dòng.
- `page_start` và `page_end` dùng cho trích dẫn.

Loader hiện tự chuyển `º` thành `o` để sửa các lỗi phổ biến như `dºanh`,
`hºặc`, `theº` và `khºản`. Những lỗi ngữ nghĩa hoặc lỗi dính từ vẫn cần được
kiểm tra trước khi đưa dữ liệu vào vector database.

## Hướng phát triển tiếp theo

1. Hoàn thiện bước hậu xử lý và đánh giá chất lượng OCR.
2. Triển khai `ingest/embedder.py`.
3. Chọn và cấu hình vector database.
4. Triển khai retriever và RAG engine.
5. Hoàn thiện FastAPI service.
6. Xây dựng bộ câu hỏi đánh giá retrieval và câu trả lời.
