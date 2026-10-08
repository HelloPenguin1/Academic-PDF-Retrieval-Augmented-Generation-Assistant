from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from backend.app.services.vision_service import DocumentProcessor
from backend.app.services.reranker import ReRanker_Model
from config.config import hf_embeddings, hf_reranker_encoder

RETRIEVER_CONFIGS = {
    1: "vector_only",
    2: "vector_bm25",
    3: "vector_rerank",
    4: "vector_bm25_rerank",
}


class Retriever:
    def __init__(self):
        self.vectorstore = None
        self.reranker: ReRanker_Model | None = None
        self.doc_processor = DocumentProcessor()
        self.docs_by_source: dict[str, list] = {}

    def load_processed_pdf(self, filepath: str, source_name: str | None = None):
        """Parse a PDF into chunks tagged with `source`. Does NOT modify the registry."""
        return self.doc_processor.load_and_process(filepath, source_name)

    def add_document(self, filepath: str, source_name: str):
        '''append all documents in a dict'''
        docs = self.load_processed_pdf(filepath, source_name)
        self.docs_by_source[source_name] = docs
        return docs

    def all_docs(self)->list:
        return [d for docs in self.docs_by_source.values() for d in docs]
    
    def list_sources(self)-> dict[str, int]:
        return {name: len(docs) for name, docs in self.docs_by_source.items()}
    
    def clear_documents(self):
        self.docs_by_source={}
        self.vectorstore = None
    
    def remove_document(self, source_name: str) -> bool:
        return self.docs_by_source.pop(source_name, None) is not None

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

        #If BM25 is involved
        if config in ("vector_bm25", "vector_bm25_rerank"):
            print("Creating BM25 retriever...")
            bm25_retriever = BM25Retriever.from_documents(docs, k=25)
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever], 
                weights=[0.4, 0.6]
            )

        #if reranker is involvedq
        if config in ("vector_rerank", "vector_bm25_rerank"):
            if self.reranker is None:
                self.reranker = ReRanker_Model(hf_reranker_encoder)
            retriever = self.reranker.create_compression_retriever(retriever)

        return retriever

    def get_statistics(self) -> dict:
        return {
            "processed_documents": len(self.all_docs()),
            "sources":self.list_sources(),
            "vectorstore_ready": self.vectorstore is not None,
        }
