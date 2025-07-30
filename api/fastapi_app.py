from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieval.retriever import get_retriever
from generation.llm_integration import get_llm
from generation.prompt_templates import RAG_PROMPT
from langchain_core.runnables import RunnablePassthrough

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

rag_chain = (
    {"context": get_retriever(), "question": RunnablePassthrough()}
    | RAG_PROMPT
    | get_llm()
)

@app.post("/query")
async def answer_query(request: QueryRequest):
    try:
        response = rag_chain.invoke(request.question)
        return {"answer": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}