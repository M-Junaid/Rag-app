import os
from chromadb import PersistentClient
from langchain_chroma import Chroma
from config.settings import settings
from data_preprocessing.embeddings import get_embedding_model

def get_vectorstore(persist=True):
    if persist:
        client = PersistentClient(path=settings.VECTOR_DB_PATH)
        return Chroma(
            client=client,
            collection_name="rag_collection",
            embedding_function=get_embedding_model()
        )
    else:
        return Chroma(
            collection_name="rag_collection",
            embedding_function=get_embedding_model()
        )

def ingest_documents(documents):
    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents)
    
