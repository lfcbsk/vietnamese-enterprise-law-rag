# vietnamese-enterprise-law-rag

Retrieval-augmented generation (RAG) system for Vietnamese enterprise law documents.

## File structure

```
vietnamese-enterprise-law-rag/
├── api/                          # FastAPI service — chat & health endpoints
│   ├── routers/
│   │   ├── chat.py               # Chat / Q&A routes
│   │   └── health.py             # Health-check routes
│   ├── schemas/
│   │   └── chat.py               # Request/response models for chat
│   ├── services/
│   │   ├── llm_client.py         # LLM API client
│   │   ├── rag_engine.py         # RAG orchestration (retrieve + generate)
│   │   └── retriever.py          # Vector / semantic search
│   ├── config.py                 # API configuration
│   ├── Dockerfile                # API container image
│   └── main.py                   # FastAPI app entry point
│
├── ingest/                       # Document ingestion pipeline
│   ├── loaders/
│   │   └── pdf_loader.py         # PDF parsing and text extraction
│   ├── embedder.py               # Embedding generation
│   ├── run.py                    # Ingestion CLI / entry script
│   └── Dockerfile                # Ingest container image
│
├── evaluation/                   # RAG quality benchmarks
│   ├── eval_answer.py            # End-to-end answer evaluation
│   ├── eval_retrieval.py         # Retrieval accuracy evaluation
│   ├── question.json             # Evaluation question set
│   └── results.md                # Evaluation results
│
├── data/                         # Source law documents (PDFs)
│   ├── 59.signed.pdf
│   └── 59tiep.pdf
│
├── docker-compose.yaml           # Local stack (API, vector DB, etc.)
├── .env.example                  # Environment variable template
├── .gitignore
├── LICENSE
└── README.md
```

### Directory overview

| Path | Purpose |
|------|---------|
| `api/` | HTTP API that answers legal questions using retrieved context and an LLM. |
| `ingest/` | Loads PDFs from `data/`, chunks text, embeds, and indexes into the vector store. |
| `evaluation/` | Scripts and datasets to measure retrieval and answer quality. |
| `data/` | Raw Vietnamese enterprise law PDFs used as the knowledge base. |
