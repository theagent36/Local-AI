# Local Agentic RAG with Vision AI

A privacy-focused, fully local Retrieval-Augmented Generation (RAG) assistant built using Streamlit, ChromaDB, and local models via Ollama. It processes digital PDFs as well as scanned or handwritten notes using multimodal vision fallback.

## Features
- **Local Document Retrieval**: ChromaDB vector store for chunked document embeddings.
- **Multimodal OCR**: Automatic fallback to `llama3.2-vision` via Ollama for handwritten or scanned PDF pages.
- **Interactive UI**: Conversational interface powered by Streamlit.

## Prerequisites
- [Python 3.10+](https://www.python.org/)
- [Ollama](https://ollama.com/) installed and running locally
- Pull the required models:
  ```bash
  ollama run llama3.2-vision
  ollama run llama3.2
  ```

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/theagent36/Local-AI.git](https://github.com/theagent36/Local-AI.git)
   cd Local-AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```
