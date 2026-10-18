"""FastAPI entrypoint: multi-document ingestion and conversational querying.

Request flow
    upload:  PDF -> DocumentProcessor (chunks) -> DocumentIndex.add (embeds only new chunks)
             -> refresh_pipeline (rebuild lightweight retriever + chain)
    query:   RAG_Pipeline.query (history-aware retrieval -> rerank -> answer)

Run from the `ResearchPro_AdvancedRAG` directory:
    uvicorn backend.app.main:app --reload
"""

import os
import shutil
import tempfile
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.services.index_service import DocumentIndex
from backend.app.services.processor_service import DocumentProcessor
from backend.app.services.rag_service import RAG_Pipeline
from backend.app.services.retriever_service import RetrieverFactory
from backend.utils.session_manager import SessionManager
from config.config import hf_embeddings, llm

RETRIEVER_KEY = 4  # vector + BM25 + rerank (see RETRIEVER_CONFIGS)


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = "default_session"


app = FastAPI(
    title="RAG Academia v2",
    description="Multi-document RAG over academic PDFs",
    version="2.1.0",
)

processor = DocumentProcessor()
index = DocumentIndex(hf_embeddings)
retriever_factory = RetrieverFactory(index)
rag_pipeline = RAG_Pipeline(llm)
session_manager = SessionManager()


def refresh_pipeline() -> None:
    """Point the RAG chain at the current index contents.

    Cheap: no embedding happens here. Embedding is done once, in DocumentIndex.add.
    """
    if index.is_empty:
        rag_pipeline.conversational_rag = None
        return

    retriever = retriever_factory.build(RETRIEVER_KEY)
    rag_chain = rag_pipeline.create_rag_chain(retriever)
    rag_pipeline.create_conversational_chain(
        rag_chain, session_manager.get_session_history
    )


def library_summary() -> dict:
    sources = index.sources()
    return {"sources": sources, "total_chunks": sum(sources.values())}


def parse_upload(file: UploadFile, source_name: str):
    """Save the upload to a temp dir (auto-cleaned) and parse it into chunks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, source_name)
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        return processor.load_and_process(path, source_name)


# -------------------------------------------------------------- endpoints


@app.get("/")
def root():
    return {
        "message": "Welcome to the Advanced Research Assistant",
        "endpoints": {
            "POST /upload_file": "Add a PDF to the collection (re-uploading a name replaces it)",
            "GET /documents": "List loaded documents and chunk counts",
            "DELETE /documents/{source_name}": "Remove one document",
            "DELETE /delete": "Remove all documents and clear chat sessions",
            "POST /query": "Ask a question across all loaded documents",
        },
    }


@app.post("/upload_file")
def upload_file(
    file: Annotated[UploadFile, File(description="PDF to add to the collection")],
):
    source_name = os.path.basename(file.filename or "")
    if not source_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        docs = parse_upload(file, source_name)
        index.add(source_name, docs)
        refresh_pipeline()
    except ValueError as e:  # e.g. PDF produced no chunks
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e!s}")

    return {
        "message": f"'{source_name}' added.",
        "stats": {"documents": len(docs), **library_summary()},
    }


@app.get("/documents")
def list_documents():
    return library_summary()


@app.delete("/documents/{source_name}")
def delete_document(source_name: str):
    if not index.remove(source_name):
        raise HTTPException(status_code=404, detail=f"'{source_name}' not found")

    refresh_pipeline()
    return {"message": f"'{source_name}' removed.", **library_summary()}


@app.delete("/delete")
def reset_all():
    """Remove every document and clear all chat sessions."""
    index.clear()
    refresh_pipeline()
    session_manager.clear_all_sessions()
    return {"message": "All documents and sessions cleared"}


@app.post("/query")
def query_rag(request: QueryRequest):
    if index.is_empty:
        raise HTTPException(
            status_code=409, detail="No documents loaded. Upload a PDF first."
        )

    return {"response": rag_pipeline.query(request.query, request.session_id)}
