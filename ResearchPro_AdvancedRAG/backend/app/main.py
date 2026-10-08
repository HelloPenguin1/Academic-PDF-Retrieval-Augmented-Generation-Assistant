import os
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.services.document_service import DocumentProcessor
from backend.app.services.rag_service import RAG_Pipeline
from backend.app.services.reranker import ReRanker_Model
from backend.utils.session_manager import SessionManager
from config.config import hf_reranker_encoder, llm


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = "default_session"


# initializing fastapi
app = FastAPI(
    title="RAG API v2",
    description="A simple RAG (Retrieval Augmented Generation) API using modularity",
    version="2.0.0",
)


# Instantiate classes
document_processor = DocumentProcessor()
rag_pipeline = RAG_Pipeline(llm)
reranker = ReRanker_Model(hf_reranker_encoder)
session_manager = SessionManager()


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Advanced Research Assistant",
        "endpoints": {
            "POST /upload_file": "Upload a document for processing",
            "POST /query": "Query the uploaded documents",
        },
    }


## API endpoint for uplaoding docs
@app.post("/upload_file")
async def upload_file(
    file: Annotated[UploadFile, File(description="Upload a text document to process")],
):
    temp_file_path = None
    try:
        if not os.path.exists("backend/temp"):
            os.makedirs("backend/temp")

        temp_file_path = os.path.join("backend/temp", f"temp_{file.filename}")
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Load and process document
        docs = document_processor.load_and_process_pdf(temp_file_path)

        # Create retrievers
        semantic_retriever = document_processor.create_retriever(docs)

        compression_retriever = reranker.create_compression_retriever(
            semantic_retriever
        )

        rag_pipeline.set_compression_retriever(compression_retriever)

        # Update vectorstore
        if document_processor.vectorstore:
            rag_pipeline.update_vectorstore(document_processor.vectorstore)
        else:
            raise HTTPException(
                status_code=500, detail="Vectorstore initialization failed"
            )

        # Create RAG chain
        rag_chain = rag_pipeline.create_rag_chain(compression_retriever)
        conversational_chain = rag_pipeline.create_conversational_chain(
            rag_chain, session_manager.get_session_history
        )
        rag_pipeline.conversational_rag = conversational_chain

        # Verify state
        if not rag_pipeline.vectorstore or not rag_pipeline.conversational_rag:
            raise HTTPException(
                status_code=500, detail="Failed to initialize RAG pipeline components"
            )

        # Cleanup
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return {
            "message": "File uploaded and retriever initialized successfully.",
            "stats": {"documents": len(docs)},
        }

    except HTTPException as he:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise he
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {e!s}")


## API endpoint for querying the retriever
@app.post("/query")
async def query_rag(query: QueryRequest):
    try:
        result = rag_pipeline.query(
            query.query, query.session_id
        )  # query is a function. since we already set conversational_rag in ragpipelien in ine 76
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e!s}")


@app.delete("/delete")
async def deletevectorstore():
    """Clear vectorstore and session state"""
    try:
        rag_pipeline.vectorstore = None
        document_processor.vectorstore = None
        rag_pipeline.compression_retriever = None
        rag_pipeline.conversational_rag = None
        session_manager.clear_all_sessions()
        return {"message": "Vectorstore and sessions cleared"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing vectorstore: {e!s}"
        )
