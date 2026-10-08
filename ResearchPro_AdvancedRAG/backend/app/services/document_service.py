from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from backend.app.services.vision_service import MultimodalProcessor
from backend.app.services.reranker import ReRanker_Model
from config.config import hf_embeddings, hf_reranker_encoder

RETRIEVER_CONFIGS = {
    1: "vector_only",
    2: "vector_bm25",
    3: "vector_rerank",
    4: "vector_bm25_rerank",
}


class DocumentProcessor:
    def __init__(self):
        self.vectorstore = None
        self.reranker: ReRanker_Model | None = None
        self.multimodal_processor = MultimodalProcessor()
        self.processed_docs = []

    def load_and_process_pdf(self, filepath: str):
        self.processed_docs = self.multimodal_processor.load_and_process(filepath)

        return self.processed_docs

    def create_retriever(self, docs, key: int = 4):
        """Build a retriever using the configuration selected by key."""
        if key not in RETRIEVER_CONFIGS:
            raise ValueError(
                f"Unknown retriever key {key}")

        config = RETRIEVER_CONFIGS[key]

        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(docs, hf_embeddings)
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 25}
        )
        retriever = vector_retriever

        if config in ("vector_bm25", "vector_bm25_rerank"):
            print("Creating BM25 retriever...")
            bm25_retriever = BM25Retriever.from_documents(docs, k=25)
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
            )

        if config in ("vector_rerank", "vector_bm25_rerank"):
            if self.reranker is None:
                self.reranker = ReRanker_Model(hf_reranker_encoder)
            retriever = self.reranker.create_compression_retriever(retriever)

        return retriever

    def get_statistics(self) -> dict:
        return {
            "processed_documents": len(self.processed_docs),
            "vectorstore_ready": self.vectorstore is not None,
        }
