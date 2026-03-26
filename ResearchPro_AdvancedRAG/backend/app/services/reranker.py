from langchain.retrievers import ContextualCompressionRetriever 
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class ReRanker_Model():
    def __init__(self, encoderModel):
        self.rerankermodel =  HuggingFaceCrossEncoder(
            model_name=encoderModel
        )
        self.compression_retriever = None
    
    def create_compression_retriever(self, retriever):
        compressor = CrossEncoderReranker(model=self.rerankermodel,
                                          top_n=5)
        self.compression_retriever =ContextualCompressionRetriever(base_compressor=compressor, 
                                                              base_retriever=retriever)
        
        return self.compression_retriever
        
