from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert assistant. Answer based ONLY on the provided context."),
    ("human", """Context: 
{context}

Question: {question}""")
])