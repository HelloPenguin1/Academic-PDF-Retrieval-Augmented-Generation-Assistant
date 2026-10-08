import os
import traceback

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from langchain.schema import Document

from config.config import chunking_tokenizer, llm_summarize


class DocumentProcessor:
    def __init__(self):
        self.llm = llm_summarize

    def load_and_process(self, filepath: str, source_name: str | None = None) -> list[Document]:
        try:
            print("Using Docling for PDF Extraction ...")

            # convert pdf to docling document
            docling_result = DocumentConverter().convert(filepath)
            docling_doc = docling_result.document

            # intialize chunker
            chunker = HybridChunker(tokenizer=chunking_tokenizer, merge_peers=True)

            # chunking
            chunks = list(chunker.chunk(dl_doc=docling_doc))

            # pass as langchain documents
            processed_langchain_docs = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.text,
                    metadata={
                        "source": source_name,
                        "headings": chunk.meta.headings,
                        "doc_items": [item.label for item in chunk.meta.doc_items],
                    },
                )
                processed_langchain_docs.append(doc)

            return processed_langchain_docs
        
        except Exception as e:
            print(traceback.format_exc())
            raise e
