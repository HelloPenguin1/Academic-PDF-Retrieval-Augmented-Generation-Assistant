"""In-memory document index: a registry of papers plus dense (FAISS) and lexical (BM25) indexes."""

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


class DocumentIndex:
    """Keeps FAISS and BM25 in sync with the set of loaded papers.

    - FAISS is updated incrementally: only new chunks are embedded.
    - BM25 is rebuilt from stored chunks on every change (no embedding, cheap).
    """

    def __init__(self, embeddings: Embeddings, bm25_k: int = 25):
        self._embeddings = embeddings
        self._bm25_k = bm25_k
        self._docs_by_source: dict[str, list[Document]] = {}
        self._faiss: FAISS | None = None
        self._bm25: BM25Retriever | None = None

    def add(self, source: str, docs: list[Document]) -> None:
        """Add a paper. Re-adding an existing source replaces it."""
        if not docs:
            raise ValueError(f"No chunks to index for '{source}'")

        self.remove(source)
        ids = self._chunk_ids(source, len(docs))

        if self._faiss is None:
            self._faiss = FAISS.from_documents(docs, self._embeddings, ids=ids)
        else:
            self._faiss.add_documents(docs, ids=ids)

        self._docs_by_source[source] = docs  # store in document registry
        self._rebuild_bm25()

    def remove(self, source: str) -> bool:
        """Remove a paper. Returns False if it wasn't loaded."""
        docs = self._docs_by_source.pop(source, None)
        if docs is None:
            return False

        if self._docs_by_source:
            self._faiss.delete(self._chunk_ids(source, len(docs)))
        else:
            self._faiss = None  # last paper removed -> back to empty state

        self._rebuild_bm25()
        return True

    def clear(self) -> None:
        self._docs_by_source.clear()
        self._faiss = None
        self._bm25 = None

    # ---------- reads ----------

    @property
    def is_empty(self) -> bool:
        return not self._docs_by_source

    def sources(self) -> dict[str, int]:
        """source name -> number of chunks."""
        return {name: len(docs) for name, docs in self._docs_by_source.items()}

    def all_docs(self) -> list[Document]:
        return [d for docs in self._docs_by_source.values() for d in docs]

    def dense_retriever(self, k: int = 25) -> BaseRetriever:
        self._require_data()
        return self._faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

    def lexical_retriever(self) -> BaseRetriever:
        self._require_data()
        return self._bm25

    # ---------- internals ----------

    def _rebuild_bm25(self) -> None:
        docs = self.all_docs()
        self._bm25 = (
            BM25Retriever.from_documents(docs, k=self._bm25_k) if docs else None
        )

    def _require_data(self) -> None:
        if self.is_empty:
            raise RuntimeError("Index is empty: add a document first")

    @staticmethod
    def _chunk_ids(source: str, n_chunks: int) -> list[str]:
        return [f"{source}::{i}" for i in range(n_chunks)]
