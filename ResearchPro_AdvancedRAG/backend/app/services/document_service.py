from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.config import hf_embeddings

from backend.app.services.vision_service import MultimodalProcessor

class DocumentProcessor:
    def __init__(self):
        self.vectorstore = None
        self.multimodal_processor = MultimodalProcessor()
        self.processed_docs = []

    def load_and_process_pdf(self, filepath: str):
        self.processed_docs = self.multimodal_processor.load_and_process(filepath)

        return self.processed_docs


    def create_retriever(self, docs):
        """
        Creates hybrid retriever: BM25 (keyword) + FAISS (semantic) via EnsembleRetriever.
        k=10 so the reranker has enough candidates to filter down to top_n=5.
        """
        # Semantic Retriever (Vector Search)
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(docs, hf_embeddings)
        semantic_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 25}
        )

        # Keyword Retriever (BM25)
        print("Creating BM25 retriever...")
        bm25_retriever = BM25Retriever.from_documents(docs, k=25)

        # Hybrid: 40% BM25, 60% semantic
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.4, 0.6]
        )

        return hybrid_retriever


    def get_statistics(self) -> dict:
        return {
            "processed_documents": len(self.processed_docs),
            "vectorstore_ready": self.vectorstore is not None
        }
