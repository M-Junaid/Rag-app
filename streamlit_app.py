import streamlit as st
import os
import warnings
from ingest import run_ingestion
from retrieval.retriever import get_retriever
from generation.llm_integration import get_llm
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

#  Disable telemetry to fix the capture() error
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Suppress warnings
warnings.filterwarnings("ignore")

#  Initialize retriever and LLM with error handling
@st.cache_resource
def initialize_components():
    try:
        retriever = get_retriever()
        llm = get_llm()
        return retriever, llm
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None

retriever, llm = initialize_components()

#  Only proceed if components are initialized
if retriever is not None and llm is not None:
    
    # Multiple RAG chain options - we'll try them in order
    def create_rag_chain_option1():
        """Option 1: Simple string prompt"""
        def format_prompt(inputs):
            context = inputs['context']
            question = inputs['question']
            
            prompt = f"""Use the following context to answer the question.

Context: {context}

Question: {question}

Answer:"""
            return prompt
        
        return (
            RunnableLambda(lambda x: {
                "context": retriever.invoke(x["question"]),
                "question": x["question"]
            })
            | RunnableLambda(format_prompt)
            | llm
            | StrOutputParser()
        )
    
    def create_rag_chain_option2():
        """Option 2: LangChain message format"""
        def format_messages(inputs):
            context = inputs['context']
            question = inputs['question']
            
            messages = [
                SystemMessage(content="Use the following context to answer the question."),
                HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")
            ]
            return messages
        
        return (
            RunnableLambda(lambda x: {
                "context": retriever.invoke(x["question"]),
                "question": x["question"]
            })
            | RunnableLambda(format_messages)
            | llm
            | StrOutputParser()
        )
    
    def create_rag_chain_option3():
        """Option 3: Direct invoke with context and question"""
        def invoke_llm_directly(inputs):
            context = inputs['context']
            question = inputs['question']
            
            # Try direct invocation with a simple string
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Different ways to invoke based on LLM type
            try:
                # Method 1: Direct string
                return llm.invoke(prompt)
            except:
                try:
                    # Method 2: With input key
                    return llm.invoke({"input": prompt})
                except:
                    # Method 3: With prompt key
                    return llm.invoke({"prompt": prompt})
        
        return (
            RunnableLambda(lambda x: {
                "context": retriever.invoke(x["question"]),
                "question": x["question"]
            })
            | RunnableLambda(invoke_llm_directly)
            | StrOutputParser()
        )
    
    # ✅ Try to create a working RAG chain
    rag_chain = None
    
    # Try different options
    for i, chain_creator in enumerate([create_rag_chain_option1, create_rag_chain_option2, create_rag_chain_option3], 1):
        try:
            rag_chain = chain_creator()
            # Test with a simple query
            test_result = rag_chain.invoke({"question": "test"})
            st.success(f"✅ RAG chain initialized successfully (Option {i})")
            break
        except Exception as e:
            st.warning(f"Option {i} failed: {str(e)}")
            continue
    
    if rag_chain is None:
        st.error("Failed to create RAG chain. Please check your LLM configuration.")
        st.stop()

    # ✅ Streamlit UI
    st.title("📄 Ask Your PDF")

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file is not None:
        # ✅ Better file handling
        temp_file_path = "temp.pdf"
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())
            
            st.success("✅ PDF uploaded successfully")
            
            if st.button("🔁 Ingest Document"):
                with st.spinner("Processing document..."):
                    try:
                        run_ingestion(temp_file_path)
                        st.success("✅ Document ingested into vector DB")
                    except Exception as e:
                        st.error(f"Error during ingestion: {str(e)}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
        except Exception as e:
            st.error(f"Error handling uploaded file: {str(e)}")

    query = st.text_input("❓ Ask a question about the document")

    if query:
        with st.spinner("💬 Generating answer..."):
            try:
                result = rag_chain.invoke({"question": query})
                st.markdown(f"**Answer:** {result}")
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                
                # Debug information
                st.expander("Debug Info").write({
                    "LLM Type": str(type(llm)),
                    "Error": str(e),
                    "Query": query
                })
else:
    st.error("Failed to initialize RAG components. Please check your configuration.")