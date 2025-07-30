from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len
    )
    return splitter.split_documents(documents)