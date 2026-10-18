"""Assembles a retriever for a given config from a DocumentIndex."""

from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever

from backend.app.services.index_service import DocumentIndex
from backend.app.services.reranker import ReRanker_Model
from config.config import hf_reranker_encoder

RETRIEVER_CONFIGS = {
    1: "vector_only",
    2: "vector_bm25",
    3: "vector_rerank",
    4: "vector_bm25_rerank",
}

BM25_CONFIGS = {"vector_bm25", "vector_bm25_rerank"}
RERANK_CONFIGS = {"vector_rerank", "vector_bm25_rerank"}

ENSEMBLE_WEIGHTS = [0.4, 0.6]  # [BM25, dense]


class RetrieverFactory:
    def __init__(self, index: DocumentIndex):
        self._index = index
        self._reranker: ReRanker_Model | None = None  # loaded once, reused

    def build(self, key: int = 4) -> BaseRetriever:
        if key not in RETRIEVER_CONFIGS:
            raise ValueError(f"Unknown retriever key {key}")
        config = RETRIEVER_CONFIGS[key]

        retriever = self._index.dense_retriever()

        if config in BM25_CONFIGS:
            retriever = EnsembleRetriever(
                retrievers=[self._index.lexical_retriever(), retriever],
                weights=ENSEMBLE_WEIGHTS,
            )

        if config in RERANK_CONFIGS:
            retriever = self._get_reranker().create_compression_retriever(retriever)

        return retriever

    def _get_reranker(self) -> ReRanker_Model:
        if self._reranker is None:
            self._reranker = ReRanker_Model(hf_reranker_encoder)
        return self._reranker
