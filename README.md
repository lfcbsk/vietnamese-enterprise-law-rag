# Vietnamese Enterprise Law RAG

Hệ thống Retrieval-Augmented Generation (RAG) để tra cứu Luật Doanh nghiệp
Việt Nam. Dự án gồm pipeline đọc/OCR tài liệu, tạo dense index bằng ChromaDB,
tạo BM25 index, hybrid retrieval, FastAPI backend và giao diện chat Streamlit.

> Câu trả lời của hệ thống chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.

## Thành phần chính

```text
src/
├── ingest/       # Đọc PDF, OCR, làm sạch và tách văn bản thành các điều luật
├── indexing/     # Sinh embedding, ChromaDB dense index và BM25 index
├── retrieval/    # Dense, BM25 và hybrid retriever
├── generation/   # Prompt và bộ dựng context
├── memory/       # Lưu trạng thái hội thoại bằng SQLite
├── api/          # FastAPI backend
└── app/          # Streamlit frontend
evaluation/       # Bộ câu hỏi, metric và script đánh giá retrieval
configs/          # Cấu hình tham khảo cho OCR
data/             # PDF nguồn, OCR cache, chunks, index và chat database
```

## Yêu cầu

- Python 3.12 (phiên bản được cố định trong `.python-version`).
- Khuyến nghị dùng [uv](https://docs.astral.sh/uv/) để cài đúng bộ dependency
  trong `uv.lock`.
- Kết nối Internet ở lần đầu để tải package và model embedding
  `intfloat/multilingual-e5-base`.
- API key Google Gemini để chạy backend hỏi đáp.
- Tesseract OCR 5.x cùng language data `vie` và `eng` chỉ bắt buộc khi muốn OCR
  lại PDF. Nếu dùng các file cache có sẵn trong `data/ocr/` thì không cần OCR lại.

Các dependency trực tiếp được cố định phiên bản trong `pyproject.toml`; toàn bộ
dependency bắc cầu được khóa chính xác trong `uv.lock` để môi trường có thể tái lập.

## 1. Cài đặt môi trường

Chạy các lệnh từ thư mục gốc của repository.

### Cách khuyến nghị: uv

```powershell
uv python install 3.12
uv sync --dev
```

`uv sync` tạo/cập nhật `.venv` và cài đúng phiên bản trong `uv.lock`. Có thể chạy
lệnh qua `uv run ...` mà không cần kích hoạt virtual environment.

Nếu muốn kích hoạt môi trường trên PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Trên macOS/Linux:

```bash
source .venv/bin/activate
```

### Cách thay thế: pip

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Với macOS/Linux, thay hai lệnh đầu bằng:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## 2. Cấu hình biến môi trường

Tạo `.env` từ file mẫu:

```powershell
Copy-Item .env.example .env
```

Trên macOS/Linux:

```bash
cp .env.example .env
```

Sau đó sửa `.env`:

```dotenv
LLM_API_KEY=your_google_gemini_api_key
LLM_MODEL=gemini-2.5-flash
LLM_BASE_URL=
LLM_TEMPERATURE=0

RAG_TOP_K=5
RAG_CANDIDATE_K=40

CHAT_DB_PATH=data/chat/checkpoints.sqlite

EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-base
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=16
CHROMA_PERSIST_DIR=data/chroma_db
CHROMA_COLLECTION_NAME=law_chunks

API_URL=http://127.0.0.1:8000
```

Ý nghĩa các biến quan trọng:

| Biến | Mục đích |
| --- | --- |
| `LLM_API_KEY` | API key dùng để gọi Gemini; bắt buộc khi chạy API |
| `LLM_MODEL` | Model sinh câu trả lời |
| `RAG_TOP_K` | Số đoạn luật cuối cùng đưa vào context |
| `RAG_CANDIDATE_K` | Số ứng viên lấy từ mỗi retriever trước khi fusion |
| `CHAT_DB_PATH` | SQLite database lưu trạng thái hội thoại |
| `EMBEDDING_MODEL_NAME` | Sentence Transformers model cho index và truy vấn |
| `EMBEDDING_DEVICE` | `cpu`, `cuda` hoặc thiết bị được PyTorch hỗ trợ |
| `EMBEDDING_BATCH_SIZE` | Batch size khi sinh embedding |
| `CHROMA_PERSIST_DIR` | Thư mục lưu persistent ChromaDB |
| `CHROMA_COLLECTION_NAME` | Tên collection chứa các điều luật |
| `API_URL` | Backend mà Streamlit sẽ gọi |

## 3. Chuẩn bị dữ liệu

Repository đã có PDF trong `data/` và OCR cache trong `data/ocr/`. Để tạo lại
`data/law_chunks.json` từ cache hiện có:

```powershell
uv run python -m src.ingest.run
```

Pipeline sẽ đọc mọi PDF trong `data/`, ưu tiên OCR cache, làm sạch văn bản, học
từ điển âm tiết từ toàn bộ corpus để tách thận trọng các từ OCR bị dính, sau đó
tách theo Chương/Điều, kiểm tra chunk rỗng hoặc ID trùng và ghi kết quả vào
`data/law_chunks.json`.

### OCR lại toàn bộ PDF

Kiểm tra Tesseract trước:

```powershell
tesseract --version
tesseract --list-langs
```

Danh sách ngôn ngữ phải có `vie` và `eng`. Sau đó chạy:

```powershell
uv run python -m src.ingest.run --force-ocr
```

Lệnh này render và OCR lại toàn bộ trang ở 300 DPI, ghi đè cache tương ứng trong
`data/ocr/`, rồi tạo lại `data/law_chunks.json`. Quá trình có thể mất nhiều thời
gian. Xem thêm tùy chọn bằng:

```powershell
uv run python -m src.ingest.run --help
```

## 4. Tạo index

Sau khi có `data/law_chunks.json`, tạo đồng thời dense index và BM25 index:

```powershell
uv run python -m src.indexing.build_indexes
```

Lần đầu Sentence Transformers sẽ tải model embedding. Kết quả chính:

- Dense index: `data/chroma_db/`.
- BM25 index: `data/indexes/bm25_index.pkl`.
- Manifest lần build: `data/indexes/manifest.json`.

Chỉ build một loại index:

```powershell
# Chỉ BM25
uv run python -m src.indexing.build_indexes --skip-dense

# Chỉ dense index
uv run python -m src.indexing.build_indexes --skip-bm25
```

Khi đổi dữ liệu nguồn, model embedding hoặc tên collection, cần build lại index
trước khi chạy API.

## 5. Chạy ứng dụng

Cần mở hai terminal tại thư mục gốc dự án.

### Terminal 1: FastAPI backend

```powershell
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Các URL hữu ích:

- Health check: <http://127.0.0.1:8000/health>
- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>

Thử API bằng PowerShell:

```powershell
$body = @{
    question = "Công ty cổ phần cần có ít nhất bao nhiêu cổ đông?"
    conversation_id = $null
} | ConvertTo-Json -Compress

# Windows PowerShell 5.1 không luôn gửi chuỗi có dấu bằng UTF-8.
$utf8Body = [System.Text.Encoding]::UTF8.GetBytes($body)

Invoke-RestMethod `
    -Method Post `
    -Uri http://127.0.0.1:8000/chat `
    -ContentType "application/json; charset=utf-8" `
    -Body $utf8Body
```

`conversation_id` có thể để trống ở câu đầu. Dùng lại ID backend trả về để tiếp
tục cùng một hội thoại.

### Terminal 2: Streamlit frontend

```powershell
uv run streamlit run src/app/app.py --server.port 8501
```

Mở <http://127.0.0.1:8501>. Frontend mặc định gọi
`http://127.0.0.1:8000`; có thể đổi bằng biến `API_URL`.

## 6. Đánh giá retrieval

Đảm bảo cả dense và BM25 index đã được tạo, sau đó chạy:

```powershell
uv run python -m evaluation.eval_retrieval --retriever hybrid
uv run python -m evaluation.eval_retrieval --retriever dense
uv run python -m evaluation.eval_retrieval --retriever bm25
```

Có thể điều chỉnh tham số, ví dụ:

```powershell
uv run python -m evaluation.eval_retrieval `
    --retriever hybrid `
    --top-k 5 `
    --candidate-k 20 `
    --dense-weight 0.9 `
    --bm25-weight 0.1
```

Kết quả được ghi vào `evaluation/outputs/retrieval_<retriever>.json`.

## 7. Kiểm tra chất lượng mã

```powershell
uv run ruff check .
uv run pytest
```

Nếu repository chưa có test, `pytest` có thể trả mã thoát 5 với thông báo không
tìm thấy test; lệnh import và Ruff vẫn dùng được để kiểm tra nhanh môi trường.

## Xử lý lỗi thường gặp

### Không tìm thấy dense index hoặc BM25 index

Chạy lại:

```powershell
uv run python -m src.indexing.build_indexes
```

### Không tải được model embedding

Kiểm tra kết nối Internet/proxy. Model được tải từ Hugging Face ở lần đầu và được
dùng lại từ cache ở những lần sau.

### `LLM_API_KEY` bị thiếu

Đảm bảo đã tạo `.env`, điền key thật và chạy Uvicorn từ thư mục gốc repository.

### Streamlit không kết nối được backend

Kiểm tra Uvicorn còn chạy, truy cập `/health`, và xác nhận `API_URL` trỏ đúng host
cùng port của API.

### Tesseract không tìm thấy `vie`

Cài thêm Vietnamese trained data, kiểm tra `TESSDATA_PREFIX` nếu Tesseract được
cài ở vị trí tùy chỉnh, rồi chạy lại `tesseract --list-langs`.

## Docker

`docker-compose.yaml` hiện chưa có nội dung và repository chưa có Dockerfile, vì
vậy dự án chưa hỗ trợ chạy bằng Docker Compose. Dùng quy trình `uv` ở trên cho đến
khi cấu hình container được bổ sung.

## CI tự động

GitHub Actions workflow tại `.github/workflows/ci.yml` tự chạy khi push hoặc mở
pull request. Workflow kiểm tra lockfile, Ruff, compile, unit test, tái tạo chunks
từ OCR cache, kiểm tra output ingestion và quality gate BM25 (`Hit@5 >= 0.65`).

Job hybrid đầy đủ chạy lúc 02:00 UTC mỗi thứ Hai (09:00 giờ Việt Nam) hoặc khi
chọn **Run workflow** trên GitHub Actions. Job này build lại ChromaDB/BM25 và yêu
cầu hybrid retrieval đạt `Hit@5 >= 0.90`.

Workflow hiện là CI kiểm tra chất lượng. Dự án chưa cấu hình bước CD/deploy vì
chưa có môi trường triển khai hoặc container image đích.
