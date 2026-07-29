# ResearchPro — Academia Multimodal RAG

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.49+-red.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq-purple.svg)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-grade **Advanced RAG (Retrieval-Augmented Generation)** system designed for academic research. Upload any research paper as a PDF and have a **context-aware, citation-grounded conversation** with it — powered by hybrid retrieval, cross-encoder reranking, and multiple specialized LLMs.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Approach](#-solution-approach)
- [Tech Stack](#-tech-stack)
- [Architecture Overview](#-architecture-overview)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Design Decisions](#-design-decisions)

---

## 🎯 Problem Statement

Academic researchers face significant pain points when working with research papers:

| Challenge | Impact |
|-----------|--------|
| **Context Loss** | Standard RAG treats PDFs as raw text, discarding headers, structure, and table formatting |
| **Query Ambiguity** | Follow-up questions like *"What about the baseline?"* fail without conversational memory |
| **Retrieval Precision** | A single retrieval method (semantic *or* keyword) leaves relevant chunks unfound |
| **Hallucination Risk** | Generic LLMs readily fabricate data, citations, and statistics |
| **Dense Academic Language** | Vague or abbreviation-heavy queries fail to retrieve the right content |

---

## 💡 Solution Approach

ResearchPro addresses these challenges with a **multi-stage, grounded RAG pipeline**:

### 1. Structure-Preserving PDF Parsing
Uses **Docling** to convert PDFs to structured Markdown, preserving headers, tables, lists, and document hierarchy. The Markdown is then split on heading boundaries — not arbitrary character counts — keeping sections semantically cohesive.

### 2. Hybrid Retrieval (BM25 + FAISS)
Every query triggers two retrievers simultaneously:
- **BM25** — exact keyword matching (40% weight)
- **FAISS** — dense semantic similarity (60% weight)

Results are fused via `EnsembleRetriever`, then the top-25 candidates from each are passed to the reranker.

### 3. Cross-Encoder Reranking
A `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder re-scores all 50 candidates as *query-document pairs*, returning only the top-5 — far more accurate than embedding-based similarity alone.

### 4. Conversational Query Reformulation
A lightweight LLM (`llama-3.3-70b-versatile`) rewrites follow-up questions into self-contained queries using the full chat history before retrieval. This means *"What about their baseline?"* becomes *"What was the baseline accuracy reported for ResNet in Table 2?"*

### 5. Faithfulness-First Answer Generation
The main LLM (`openai/gpt-oss-120b`) operates under strict system-prompt rules:
- Ground **every claim** exclusively in the retrieved context
- Refuse to speculate when context is insufficient
- Explicitly separate facts from different documents
- Answer concisely — match response complexity to question complexity

### 6. Session-Scoped Conversational Memory
Each browser tab generates a unique UUID session. Chat histories are stored in-memory on the FastAPI server and injected into every retrieval and generation step.

---

## 🛠 Tech Stack

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI Framework | **Streamlit 1.49** | Chat interface with sidebar controls |
| Session Identity | **UUID** | Per-tab session isolation |
| HTTP Client | **Requests** | REST calls to the FastAPI backend |

### Backend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | **FastAPI 0.116** | Async REST API (upload, query, delete) |
| RAG Orchestration | **LangChain 0.3** | Chain composition and history management |
| PDF Parsing | **Docling** | Structured PDF → Markdown conversion |
| Chunking | **MarkdownHeaderTextSplitter** | Header-aware semantic chunking |
| Vector Database | **FAISS (CPU)** | In-memory dense vector search |
| Keyword Search | **BM25Retriever** (rank-bm25) | Sparse keyword retrieval |
| Reranker | **CrossEncoderReranker** | Query-document pair scoring |
| Embeddings | **BAAI/bge-small-en-v1.5** | HuggingFace sentence embeddings |
| Tokenizer | **all-MiniLM-L6-v2** | Tokenization for chunking |

### AI Models

| Role | Provider | Model | Usage |
|------|----------|-------|-------|
| **Answer Generation** | Groq | `openai/gpt-oss-120b` | Final grounded Q&A |
| **Query Reformulation** | Groq | `llama-3.3-70b-versatile` | Context-aware query rewrite |
| **Summarization** (optional) | Groq | `llama-3.3-70b-versatile` | On-demand chunk summarization |
| **Embeddings** | HuggingFace | `BAAI/bge-small-en-v1.5` | Dense vector encoding |
| **Reranker** | HuggingFace | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Candidate reranking |


### RAG Query Flow (per request)

```
User Query
    │
    ▼
[Session Manager] ─── load chat history
    │
    ▼
[Query Reformulation LLM] ─── rewrite query using history
    │
    ▼
[Hybrid Retriever]
    ├── BM25 (keyword, k=25)
    └── FAISS (semantic, k=25)
         │
         └──[EnsembleRetriever, weights: 0.4 BM25 / 0.6 FAISS]
              │
              ▼
         [CrossEncoder Reranker, top_n=5]
              │
              ▼
         [Answer Generation LLM] ─── strict faithfulness prompt
              │
              ▼
         Response + Session History Update
```

---

## 📁 Project Structure

```
ResearchPro_AdvancedRAG/
│
├── backend/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app & all API endpoints
│   │   ├── evaluation/               # Evaluation scripts & results
│   │   │   ├── single_doc_eval/
│   │   │   ├── multi_doc_eval/
│   │   │   └── papers/
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── document_service.py   # DocumentProcessor — FAISS + BM25 hybrid
│   │       ├── vision_service.py     # MultimodalProcessor — Docling PDF → Markdown
│   │       ├── rag_service.py        # RAG_Pipeline — chains, prompts, query()
│   │       └── reranker.py           # ReRanker_Model — cross-encoder compression
│   └── utils/
│       ├── __init__.py
│       └── session_manager.py        # In-memory chat history store (per session UUID)
│
├── config/
│   ├── __init__.py
│   └── config.py                     # All LLM, embedding, and tokenizer initialization
│
├── frontend/
│   ├── __init__.py
│   └── streamlit_app.py              # Streamlit chat UI + sidebar controls
│
├── .env                              # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## 📡 API Reference

**Base URL**: `http://127.0.0.1:8000`

### `GET /`
Health check — returns available endpoints.

---

### `POST /upload_file`
Upload and process a PDF research paper.

**Request**: `multipart/form-data`
- `file` — PDF file (required)

**Processing Steps**:
1. Save to `backend/temp/`
2. **Docling** converts PDF → structured Markdown
3. **MarkdownHeaderTextSplitter** chunks by `#` / `##` / `###` headers
4. **FAISS** vectorstore created from chunks + `BAAI/bge-small-en-v1.5` embeddings
5. **BM25Retriever** created from same chunks
6. **EnsembleRetriever** fuses BM25 (40%) + FAISS (60%), k=25 each
7. **CrossEncoderReranker** wraps the hybrid retriever, top_n=5
8. **Conversational RAG chain** initialized (reformulation + answer prompts)
9. Temp file deleted

**Response (200)**:
```json
{
  "message": "File uploaded and retriever initialized successfully.",
  "stats": { "documents": 84 }
}
```

---

### `POST /query`
Query the processed document conversationally.

**Request**: `application/json`
```json
{
  "query": "What accuracy did the proposed method achieve on ImageNet?",
  "session_id": "optional-uuid-string"
}
```

**Processing Steps**:
1. Load session chat history
2. **Query reformulation** via `llm_reformulate` (LLM call #1)
3. **Hybrid retrieval** → reranking → top-5 chunks
4. **Answer generation** via `llm` with faithfulness prompt (LLM call #2)
5. Save updated history to session store

**Response (200)**:
```json
{
  "response": "The proposed method achieved 92.1% top-1 accuracy on ImageNet..."
}
```

---

### `DELETE /delete`
Reset the pipeline — clears vectorstore, retrievers, chains, and all session histories.

**Response (200)**:
```json
{ "message": "Vectorstore and sessions cleared" }
```

---

## 🧠 Design Decisions

### Why Docling over PyPDF / pdfminer?
Docling understands document *structure* — it preserves heading hierarchies, table formatting, and list nesting. This means chunks stay semantically coherent instead of being arbitrary text windows that break mid-sentence or mid-table.

### Why Markdown Header Splitting?
Splitting on `#`/`##`/`###` headers respects the paper's logical structure (Introduction, Methods, Results, etc.), ensuring retrieved chunks correspond to complete argumentative units rather than fragments.

### Why BM25 + FAISS instead of FAISS alone?
Dense embeddings struggle with exact term matching — model names, dataset names, and numeric values. BM25 catches these precisely. The ensemble fuses the strengths of both: conceptual similarity *and* exact-term recall.

### Why a Cross-Encoder Reranker?
Bi-encoder similarity (used in FAISS) approximates relevance independently for query and document. A cross-encoder sees the *pair together*, giving much higher-quality relevance scores. Running it on 50 pre-filtered candidates keeps latency low while maximizing precision.

### Why two separate LLMs (reformulation vs. answer)?
Query reformulation requires only a short output (a rewritten query) and benefits from lower latency. Using a separate, lighter call for this keeps the expensive main LLM focused entirely on answer quality. It also avoids hitting rate limits on a single model.

### Why session UUIDs from the frontend?
Each Streamlit session generates its own UUID, allowing multiple concurrent users to maintain isolated chat histories on the same server instance without any authentication infrastructure.

---

## 🗺 Roadmap

- [ ] Multi-document support (cross-paper queries)
- [ ] Persistent vectorstore (disk-backed FAISS or ChromaDB)
- [ ] Streaming responses via WebSocket or SSE
- [ ] RAGAS-based automated evaluation pipeline
- [ ] Docker Compose setup for one-command deployment
- [ ] Vision model integration for chart/figure understanding

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
