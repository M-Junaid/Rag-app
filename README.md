# RAG-App

A Retrieval-Augmented Generation (RAG) application using FastAPI for backend APIs and Streamlit for the frontend. The app supports document ingestion, chunking, embedding, vector storage, and LLM-based answer generation.

## Features

- **Document Ingestion:** Upload and process documents (PDF, TXT, etc.)
- **Chunking & Embedding:** Preprocess documents and generate embeddings.
- **Vector Store:** Store and retrieve document embeddings.
- **Retrieval:** Retrieve relevant chunks using vector similarity.
- **LLM Integration:** Generate answers using a language model.
- **API:** FastAPI backend for programmatic access.
- **Frontend:** Streamlit app for user interaction.

## Project Structure

```
rag-app/
│
├── api/                   # FastAPI backend
├── config/                # Configuration files
├── data/                  # Data files and vector stores
├── data_preprocessing/    # Chunking, loading, embedding scripts
├── generation/            # LLM integration and prompt templates
├── retrieval/             # Retrieval and vector DB logic
├── tests/                 # Unit tests
├── ingest.py              # Data ingestion script
├── streamlit_app.py       # Streamlit frontend
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd rag-app
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI backend:**
   ```bash
   uvicorn api.fastapi_app:app --reload
   ```

5. **Run the Streamlit frontend:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

- Use the Streamlit app to upload documents and ask questions.
- The backend handles document processing, embedding, and retrieval.

## Testing

Run unit tests with:
```bash
pytest tests/
```

## Notes

- Do **not** commit the `venv/` folder or large data files to version control.
- Add sensitive information (API keys, etc.) to environment variables or a `.env` file (not included in repo).

## License

[MIT License](LICENSE) (or your preferred license)
