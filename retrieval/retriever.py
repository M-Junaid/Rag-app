from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from generation.llm_integration import get_llm
from retrieval.vector_db import get_vectorstore
from config.settings import settings  # ✅ required import

def get_retriever():
    vectorstore = get_vectorstore()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K})
    
    compressor = LLMChainExtractor.from_llm(get_llm())
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
