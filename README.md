# OCR and Retrieval-Augmented Chatbot for Vietnamese Documents

A self-hosted system that converts scanned and digital documents into searchable knowledge and answers questions about them in Vietnamese. It combines an asynchronous OCR ingestion pipeline with a Retrieval-Augmented Generation (RAG) chatbot. All models run on infrastructure you control; no document content is sent to external services.

## Table of Contents

1. [Overview](#1-overview)
2. [Technology Stack](#2-technology-stack)
3. [System Architecture](#3-system-architecture)
4. [Input](#4-input)
5. [Pipeline](#5-pipeline)
6. [Expected Outcomes](#6-expected-outcomes)
7. [API Reference](#7-api-reference)
8. [Configuration](#8-configuration)
9. [Deployment](#9-deployment)
10. [Testing](#10-testing)
11. [Project Structure](#11-project-structure)
12. [Operational Notes and Limitations](#12-operational-notes-and-limitations)

## 1. Overview

The system consists of two cooperating modules:

- **Ingestion (`app/ocr`)** accepts documents from S3-compatible object storage, extracts their text, corrects spelling, masks sensitive terms, stores the result in PostgreSQL, and indexes the content in a vector database.
- **Chatbot (`app/embedding`)** answers user questions by retrieving the most relevant passages, re-ranking them, and generating a grounded answer with a local large language model. It also keeps short-term conversation memory and can summarize a document on request.

Both modules are exposed through authenticated REST APIs and share a single Celery application for background work.

## 2. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| API framework | FastAPI, Uvicorn, Pydantic | REST endpoints, request validation, HTTP Basic authentication |
| Task orchestration | Celery, Celery Beat | Multi-stage asynchronous pipeline and scheduled jobs |
| Message broker | RabbitMQ | Task queues |
| Relational database | PostgreSQL (psycopg2) | Celery result backend, document registry, chat history |
| Vector database | Milvus (pymilvus), HNSW index, inner-product metric | Storage and similarity search of document chunks |
| Object storage | MinIO or any S3-compatible service (boto3) | Source documents |
| Model serving | vLLM (OpenAI-compatible HTTP API) | Hosting the OCR model and the answer-generation model |
| OCR model | `datalab-to/chandra-ocr-2` | Full-page text recognition from page images |
| Spelling correction | `yammdd/vietnamese-error-correction` (BARTpho-based, Hugging Face Transformers) | Correcting OCR errors in Vietnamese text |
| Embedding model | `AITeamVN/Vietnamese_Embedding` (Sentence-Transformers, 1024 dimensions) | Dense vector representation of chunks and queries |
| Re-ranking model | `AITeamVN/Vietnamese_Reranker` (cross-encoder) | Precise relevance scoring of retrieved candidates |
| Answer generation | `AITeamVN/GRPO-VI-Qwen2-7B-RAG` | Grounded answer and summary generation |
| Image processing | OpenCV, Pillow, pdf2image (Poppler) | Rendering, contrast enhancement, deskew, resizing |
| Document parsing | python-docx, LibreOffice (headless) | Direct text extraction from `.docx` and `.doc` |
| Text splitting | LangChain Text Splitters | Overlapping chunking |
| Deployment | Docker Compose, systemd | Infrastructure services and model servers |

## 3. System Architecture

```
                         +-----------------------------+
                         |  S3-compatible storage      |
                         +--------------+--------------+
                                        |
        +-------------------------------v-------------------------------+
        |  OCR API (port 9090)                Celery Beat                |
        |  submit / status / documents        periodic bucket sync       |
        +---------------+---------------------------+-------------------+
                        |                           |
                        v                           v
                 +--------------+            +-------------+
                 |  RabbitMQ    |  <-------  |  Workers    |
                 +--------------+            +------+------+
                                                    |
        +---------------------+---------------------+-------------------+
        |                     |                     |                   |
        v                     v                     v                   v
  vLLM: Chandra OCR     BARTpho (local)       PostgreSQL            Milvus
  (port 8101)           spelling model        document registry     vector index
                                              chat history
                                                                        ^
  +---------------------------------------------------------------------+
  |
  |   Chatbot API (port 9091) --> embedding model --> Milvus search
  |                           --> reranker model
  |                           --> vLLM: answer model (port 8114)
```

Default service ports:

| Service | Port |
|---|---|
| OCR API | 9090 |
| Chatbot API | 9091 |
| vLLM, OCR model | 8101 |
| vLLM, answer model | 8114 |
| Milvus | 19530 |
| PostgreSQL | 5432 |
| RabbitMQ | 5672 (management UI 15672) |

## 4. Input

### 4.1 Documents for ingestion

| Category | Accepted formats | Processing path |
|---|---|---|
| Scanned documents and images | `.pdf`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`, `.gif` | Image preprocessing followed by OCR |
| Digital documents | `.docx`, `.doc`, `.txt` | Direct text extraction, OCR is skipped |

Documents can enter the system in two ways:

1. **Explicit submission.** A client calls `POST /api/submit-ocr` with an S3 path and credentials for that storage:

   | Field | Type | Description |
   |---|---|---|
   | `s3_path` | string | Location of the file, for example `s3://bucket/path/file.pdf` |
   | `access_key` | string | S3-compatible access key |
   | `secret_key` | string | S3-compatible secret key |
   | `endpoint` | string | S3-compatible endpoint URL |
   | `document_id` | integer, optional | Identifier from the calling system |

2. **Automatic bucket synchronization.** When `MINIO_SYNC_BUCKET` is configured, a scheduled job compares the bucket with the document registry and processes new, modified, and deleted files without any client action.

### 4.2 Chat requests

| Field | Type | Description |
|---|---|---|
| `question` | string, 1 to 2000 characters | The user question, in Vietnamese |
| `session_id` | string, up to 128 characters, optional | Identifies a conversation. If omitted, the server creates one and returns it |

Examples of supported intents: factual questions about document content, follow-up questions that rely on earlier turns, and summarization requests such as "Summarize this document".

### 4.3 Authentication

All endpoints except `/health` require HTTP Basic authentication. The credentials are taken from `API_USERNAME` and `API_PASSWORD`. If either is empty, every request is rejected.

## 5. Pipeline

### 5.1 Ingestion pipeline

Each document passes through four chained Celery tasks. Each task runs on a dedicated queue so that CPU-bound, GPU-bound, and database work can be scaled independently.

```
prepare_ocr_task -> ocr_task -> save_result_task -> chunk_embed_task
 (prepare_queue)  (process_queue)   (db_queue)        (embed_queue)
```

**Stage 1: Preparation (`prepare_queue`, CPU and I/O bound)**

1. Download the object from S3 and record its ETag.
2. Validate the content type and file extension.
3. Read PDF metadata (creator and creation date) when available.
4. Branch by file type:
   - `.txt`, `.docx`, `.doc`: extract text directly. For `.docx`, paragraphs and tables are read in true document order and merged cells are de-duplicated. `.doc` files are converted with headless LibreOffice first.
   - PDF and images: render PDF pages at 200 DPI, convert to grayscale, apply CLAHE contrast enhancement, correct skew using a projection-profile estimate, limit the long side to 5800 pixels, and encode each page as JPEG.

**Stage 2: Recognition and post-processing (`process_queue`, GPU bound)**

1. Send each page image to the Chandra OCR model through vLLM. Pages of one document are processed in parallel, one request per page, so the total number of pages does not grow the per-request token budget.
2. Remove the structured wrapper and HTML markup from the model output and join pages with an explicit page-break marker.
3. Correct spelling with the BARTpho model. Text is split into lines of at most 40 words and processed in batches. If a batch fails, the original text is kept for that batch.
4. Mask sensitive terms using the list in `app/ocr/sensitive_words.txt`. Longer phrases take priority over single words, matching tolerates irregular whitespace, and each masked term is replaced by asterisks of equal length.
5. Direct-extraction documents skip OCR and spelling correction but still receive sensitive-term masking.

**Stage 3: Persistence (`db_queue`)**

The result is upserted into the `ocr_documents` table with its status (`processing`, `success`, `empty`, or `error`), the raw text, the corrected text, and metadata.

**Stage 4: Chunking and embedding (`embed_queue`)**

1. Concatenate the full document text and split it with a recursive splitter (default 1000 characters with 150 characters of overlap). Splitting is applied across page boundaries so that overlap bridges pages; each chunk still records its page index.
2. Encode chunks with the embedding model into normalized 1024-dimensional vectors.
3. Replace any existing chunks of the same document in Milvus with the new ones.

**Bucket synchronization (Celery Beat)**

`sync_minio_task` runs on a fixed interval, lists the configured bucket, and compares each object's ETag with the document registry. New or changed objects enter the pipeline above. Objects that no longer exist are removed from both PostgreSQL and Milvus. `cleanup_chat_history_task` runs on its own interval to delete expired conversation records.

### 5.2 Question-answering pipeline

```
question -> retrieval query -> embedding -> vector search -> re-ranking
         -> filtering and de-duplication -> prompt assembly -> answer model
         -> answer with sources
```

1. **Retrieval query.** The current question is combined with the most recent earlier user questions from the same session, so that follow-up questions containing pronouns still retrieve the right passages. Only the original question is passed to the answer model.
2. **Vector search.** The query is embedded and the top candidates are retrieved from Milvus (default 10).
3. **Re-ranking.** A cross-encoder rescores every candidate against the query.
4. **Filtering.** Candidates below a minimum score are dropped, duplicate passages are removed, and the best passages are kept (default 5). If nothing remains, the system answers that no relevant document was found instead of guessing.
5. **Generation.** The selected passages, the last conversation turns (default 3), and the question are sent to the answer model through vLLM. Internal reasoning markup is stripped from the output.
6. **Response.** The answer is returned together with the sources used: bucket, object key, chunk index, and relevance score.

**Conversation memory.** Turns are stored in PostgreSQL per session. A session expires after a period of inactivity (default 90 minutes), after which earlier turns are ignored and eventually deleted.

**Document summarization.** Requests containing phrases such as "tóm tắt" are routed to a separate path. The target document is resolved from the conversation ("this document" refers to the document used in the previous turn) or from retrieval. Only the first pages of the document (default 3) are read, within a fixed character budget, and the reply states which file was summarized and whether the summary covers only part of the document.

**Concurrency control.** The number of chat requests that use the GPU at the same time is limited by `CHAT_GPU_CONCURRENCY`; additional requests wait in a queue.

## 6. Expected Outcomes

### 6.1 Ingestion

| Outcome | Description |
|---|---|
| Document record | One row per document in `ocr_documents` with status, ETag, raw OCR text, corrected text, and error details when applicable |
| Searchable index | The document's chunks and vectors are present in Milvus; on re-ingestion the previous chunks of that document are deleted and the new ones inserted |
| Clean text | Common OCR errors corrected and configured sensitive terms masked in the stored corrected text |
| Consistency with storage | New and modified files are ingested automatically; deleted files disappear from search results |
| Observable progress | Clients can poll the status of a submitted job and list processed documents |

### 6.2 Chatbot

| Outcome | Description |
|---|---|
| Grounded answers | Answers are based on retrieved passages and accompanied by source references |
| Safe refusal | When no passage reaches the relevance threshold, the system states that no relevant document was found |
| Contextual follow-ups | Follow-up questions in the same session are interpreted with the preceding turns |
| Summaries | A short summary of the first pages of a named or previously discussed document, labelled with the file name |
| Controlled access | Unauthenticated requests are rejected with HTTP 401 |

### 6.3 Indicative behavior

A successful ingestion followed by the question "What is Decision 1780/QD-TTg about?" returns a Vietnamese answer drawn from the indexed text of that decision, with a source list pointing to its chunks. A subsequent message "Summarize this document" returns a summary of the same file.

## 7. API Reference

### OCR API (port 9090)

| Method | Path | Description |
|---|---|---|
| POST | `/api/submit-ocr` | Submit a document; returns `202` with a `request_id` |
| GET | `/api/request-status/{request_id}` | Status and result of a submitted job |
| GET | `/api/documents` | List processed documents; supports `bucket`, `status`, and `limit` filters |
| POST | `/api/sync-minio` | Trigger an immediate bucket synchronization |
| GET | `/health` | Liveness check (no authentication) |

### Chatbot API (port 9091)

| Method | Path | Description |
|---|---|---|
| POST | `/api/chat` | Ask a question; returns `answer`, `sources`, and `session_id` |
| GET | `/health` | Liveness check (no authentication) |

Example:

```bash
curl -u "$API_USERNAME:$API_PASSWORD" \
  -X POST http://localhost:9091/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Decision 1780/QD-TTg regulate?", "session_id": "demo-1"}'
```

```bash
curl -u "$API_USERNAME:$API_PASSWORD" \
  -X POST http://localhost:9090/api/submit-ocr \
  -H "Content-Type: application/json" \
  -d '{"s3_path": "s3://bucket/file.pdf", "access_key": "...", "secret_key": "...", "endpoint": "http://storage:9000"}'
```

## 8. Configuration

All settings are read from the `.env` file in the project root. Replace every placeholder before running.

| Group | Variables |
|---|---|
| Messaging and database | `CELERY_RABBITMQ`, `CELERY_POSTGRES`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PW` |
| Object storage | `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_REGION`, `S3_SSL` |
| Bucket synchronization | `MINIO_SYNC_BUCKET`, `MINIO_SYNC_PREFIX`, `MINIO_SYNC_INTERVAL_SECONDS` |
| OCR model | `CHANDRA_MODEL`, `CHANDRA_URL`, `CHANDRA_MAX_TOKENS` |
| Spelling model | `SPELL_MODEL_PATH` |
| Embedding and re-ranking | `EMBEDDING_MODEL_PATH`, `EMBEDDING_DIM`, `EMBEDDING_BATCH_SIZE`, `RERANKER_MODEL_PATH` |
| Answer model | `RAG_LLM_MODEL`, `RAG_LLM_URL` |
| Vector store | `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION` |
| Chunking | `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Retrieval | `RETRIEVAL_TOP_K`, `RERANK_TOP_N`, `RERANK_MIN_SCORE`, `RETRIEVAL_HISTORY_QUESTIONS` |
| Summarization | `SUMMARY_MAX_PAGES`, `SUMMARY_MAX_CHARS`, `SUMMARY_MAX_TOKENS` |
| Conversation | `CHAT_HISTORY_TURNS`, `CHAT_SESSION_TIMEOUT_MINUTES`, `CHAT_HISTORY_CLEANUP_INTERVAL_SECONDS`, `CHAT_GPU_CONCURRENCY` |
| Security | `AUTH_ENABLED`, `API_USERNAME`, `API_PASSWORD`, `CORS_ORIGINS` |
| Startup and GPU placement | `PRELOAD_MODELS`, `OCR_GPU_ID`, `CHAT_GPU_ID` |

The `RERANK_MIN_SCORE` threshold depends on the re-ranker's score scale and should be tuned against real queries.

## 9. Deployment

### 9.1 Hardware Requirements

The reference configuration is a single Linux host with the following resources.

| Resource | Requirement |
|---|---|
| System memory | 16 GB RAM |
| GPU | 2 x NVIDIA GeForce RTX 3090, 24 GB VRAM each |
| GPU allocation | One GPU dedicated to OCR, one GPU dedicated to the chatbot |

GPU allocation:

| GPU | Role | Workloads placed on it |
|---|---|---|
| GPU 0 | OCR | vLLM server for the OCR model (port 8101); spelling-correction model loaded by the `process_queue` worker |
| GPU 1 | Chatbot | vLLM server for the answer model (port 8114); embedding and re-ranking models loaded by the Chatbot API; embedding model loaded by the `embed_queue` worker |

The assignment is controlled in three places, which must stay consistent:

- `OCR_GPU_ID` and `CHAT_GPU_ID` in `.env` (defaults `0` and `1`) are read by `start.sh`, which sets `CUDA_VISIBLE_DEVICES` for the Chatbot API and for the `process_queue` and `embed_queue` workers.
- `service/vllm-chandra.service` sets `CUDA_VISIBLE_DEVICES=0`.
- `service/llm-chatbot.service` sets `CUDA_VISIBLE_DEVICES=1`.

Memory notes:

- Every process that loads a model keeps its own copy in host RAM and in VRAM. With 16 GB of RAM, keep the worker counts in `start.sh` low; the defaults run one worker each for `process_queue` and `embed_queue`. The Chatbot API and the `embed_queue` worker each load their own copy of the embedding model on GPU 1.
- Each vLLM server is limited by `--gpu-memory-utilization` in its systemd unit. On GPU 1 the remaining VRAM must hold the embedding and re-ranking models, so lower the answer model's value if memory becomes tight. On GPU 0 the value can be raised if the spelling model leaves headroom.
- The infrastructure containers (RabbitMQ, PostgreSQL, Milvus with etcd and MinIO) and the vLLM processes also consume host RAM. Verify total usage under load before increasing worker counts.

### 9.2 Software Prerequisites

- Linux with an NVIDIA driver that supports CUDA.
- Python virtual environment with the dependencies from `requirements.txt`. If the default PyTorch wheel does not match your driver, install `torch` and `torchvision` from the matching PyTorch index first, because the two releases must match.
- `requirements.txt` also lists libraries that the current code does not import (for example Flower, SQLAlchemy, asyncpg, Alembic, bitsandbytes, and OpenAI). They are kept for future use and for operational tooling, and can be removed to shorten installation.
- vLLM installed in a separate virtual environment, referenced by the systemd units.
- Docker and Docker Compose.
- Poppler utilities (for PDF rendering) and LibreOffice (only if `.doc` files are expected).
- Local copies of the models listed in section 2.

### 9.3 Steps

1. Edit `.env` and replace all placeholder paths, hosts, and credentials.
2. Start infrastructure (RabbitMQ, PostgreSQL, Milvus with etcd and MinIO):

   ```bash
   docker compose up -d
   ```

3. Install and start the model servers. Adjust paths in `service/vllm-chandra.service` and `service/llm-chatbot.service`, then:

   ```bash
   sudo cp service/*.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now vllm-chandra llm-chatbot
   ```

4. Set `VENV_DIR` in `start.sh`, then start the application:

   ```bash
   bash start.sh
   ```

   This launches both APIs, one worker group per queue, and Celery Beat. Logs are written under `logs/`.

5. Optionally create the Milvus collection ahead of time:

   ```bash
   python -m app.embedding.init_milvus --name document_chunks
   ```

### 9.4 Stopping

```bash
bash stop.sh
```

`stop.sh` shuts the application down in this order:

1. Celery Beat, so that no new scheduled work is queued.
2. The OCR API and the Chatbot API, so that no new requests are accepted.
3. All Celery workers.

Each process receives SIGTERM first. For Celery, only the main worker process is signalled so that it can finish its current task and stop its pool cleanly. If a group is still running after `STOP_TIMEOUT` seconds (default 60), the remaining processes are killed with SIGKILL; because `task_acks_late` is enabled, an interrupted task is delivered again when the workers start. Increase the timeout for long OCR jobs, for example `STOP_TIMEOUT=300 bash stop.sh`.

The script only acts on processes that belong to the project directory it is located in, so another copy of the application running on the same host is not affected. It does not stop the infrastructure containers or the model servers:

```bash
docker compose stop
sudo systemctl stop vllm-chandra llm-chatbot
```

`stop.sh` requires Linux, because it identifies processes through `/proc`.

## 10. Testing

Tests use the standard library `unittest` only. Run them from the `test` directory.

Offline unit tests (no GPU, Milvus, or model servers required):

```bash
python -m unittest test_auth test_ocr.ProcessFileTempDirTest test_ocr.SensitiveMaskTest \
  test_embedding.RetrieveLogicTest test_embedding.BuildRetrievalQueryTest \
  test_embedding.AnswerQuestionFlowTest test_embedding.SummaryIntentTest \
  test_embedding.MergeChunkTextsTest test_embedding.SummaryFlowTest -v
```

End-to-end scripts on local files (require the running services):

| Command | Purpose |
|---|---|
| `python test_ocr.py [data_dir]` | Process every supported file in `test/data` and write one `.txt` per file to `test/output` |
| `python test_embedding.py --chat --debug` | Index the `.txt` files into the `test_embedding` collection and open an interactive chat session; `--skip-ingest` reuses existing data |
| `python test_pipeline.py --chat` | Run OCR, indexing, and chat in one command |

Test indexing uses a separate collection (`test_embedding`) so production data is not affected. When running these scripts directly, select the GPU explicitly so they follow the allocation in section 9.1, for example `CUDA_VISIBLE_DEVICES=0 python test_ocr.py` for OCR and `CUDA_VISIBLE_DEVICES=1 python test_embedding.py --chat` for the chatbot.

## 11. Project Structure (currently)

```
.
|-- app/
|   |-- auth.py                    HTTP Basic authentication dependency
|   |-- ocr/                       Ingestion module
|   |   |-- main.py                OCR REST API
|   |   |-- celery_app.py          Celery configuration, queues, schedules
|   |   |-- task.py                prepare, OCR, and persistence tasks
|   |   |-- sync_task.py           Bucket synchronization task
|   |   |-- flow.py                Preprocessing, OCR calls, post-processing
|   |   |-- spelling_service.py    Spelling-correction model wrapper
|   |   |-- utils.py               S3 helpers, document extraction, text utilities
|   |   |-- db.py                  Document registry access
|   |   |-- models.py              Request and response schemas
|   |   `-- sensitive_words.txt    Terms to mask
|   `-- embedding/                 Chatbot module
|       |-- main.py                Chatbot REST API
|       |-- chat_service.py        Retrieval, generation, summarization
|       |-- chunking.py            Text splitting with page tracking
|       |-- embedding_service.py   Embedding model wrapper
|       |-- reranker_service.py    Re-ranking model wrapper
|       |-- milvus_store.py        Vector store access
|       |-- history_db.py          Conversation history storage
|       |-- task.py                Chunk-and-embed and cleanup tasks
|       `-- init_milvus.py         Collection creation utility
|-- service/                       systemd units for the vLLM servers
|-- test/                          Unit tests and end-to-end scripts
|-- docker-compose.yml             RabbitMQ, PostgreSQL, Milvus stack
|-- start.sh                       Application launcher
|-- stop.sh                        Graceful shutdown of the application processes
`-- requirements.txt
```

## 12. Operational Notes and Limitations

- **GPU memory.** Workloads are split across two GPUs as described in section 9.1. Each GPU is shared by a vLLM server and by models loaded in application processes, so the sum of `--gpu-memory-utilization` and those models must stay within 24 GB. On a host with only one GPU, the two vLLM servers cannot both run at high memory utilization; lower `--gpu-memory-utilization` for each or run them alternately.
- **Transport security.** HTTP Basic authentication sends credentials with every request. Place the APIs behind an HTTPS reverse proxy before exposing them beyond a trusted network, and restrict `CORS_ORIGINS` to known origins.
- **OCR quality.** Recognition accuracy depends on scan quality. A page with extremely dense text can exceed `CHANDRA_MAX_TOKENS` and be truncated. If the spelling model fails at runtime, the pipeline keeps the original text and still reports success.
- **Sensitive-term masking** applies to the corrected text only; the stored raw OCR text is not masked. The term list is a plain text file that you maintain.
- **Summaries** cover only the first pages of a document and are generated automatically; verify important details against the source.
- **Fixed-size chunking** may split a clause across two chunks. Overlap mitigates this but does not remove it.
- **Credential brute force.** There is no rate limiting on authentication attempts; use a strong password.
