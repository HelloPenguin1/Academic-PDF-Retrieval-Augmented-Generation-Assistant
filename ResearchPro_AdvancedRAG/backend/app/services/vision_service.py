# from unstructured.partition.pdf import partition_pdf
# from unstructured.chunking.title import chunk_by_title
from docling.document_converter import DocumentConverter
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
#from config.config import llm_summarize 
#from docling.chunking import HybridChunker
from langchain.text_splitter import MarkdownHeaderTextSplitter


class MultimodalProcessor:
    # def __init__(self):
    #     self.llm = llm_summarize   
        
    def load_and_process(self, filepath:str) -> list[Document]:
        try:
            print("Using Docling for PDF Extraction ...")
            
            #convert pdf to docling document
            docling_result = DocumentConverter().convert(filepath)
            docling_doc = docling_result.document 
            
            # #intialize chunker 
            # chunker = HybridChunker(
            #     tokenizer=chunking_tokenizer,
            #     merge_peers = False) #to not exceed token limit
            
            # #chunking 
            # chunks = list(chunker.chunk(dl_doc=docling_doc))
            markdown_text = docling_doc.export_to_markdown()
            
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            chunks = splitter.split_text(markdown_text)
            
            
            #pass as langchain documents
            processed_langchain_docs = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.page_content,
                    metadata=chunk.metadata
                )
                processed_langchain_docs.append(doc)
            
            return processed_langchain_docs
        
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise e
   
  