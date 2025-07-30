# ingest.py

from data_preprocessing.document_loader import load_documents
from data_preprocessing.chunking import chunk_documents
from retrieval.vector_db import ingest_documents

import tkinter as tk
from tkinter import filedialog

def choose_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF files", "*.pdf")]
    )
    return file_path

def run_ingestion(file_path: str):
    print(f"🔄 Loading documents from {file_path}")
    docs = load_documents(file_path)
    
    print(f"✂️ Chunking documents (total pages: {len(docs)})")
    chunks = chunk_documents(docs)
    
    print(f"📥 Ingesting {len(chunks)} chunks into vector DB")
    ingest_documents(chunks)
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    selected_file = choose_file()
    if selected_file:
        run_ingestion(selected_file)
    else:
        print("❌ No file selected.")
