# RAG Chatbot

A simple, beginner-friendly Retrieval-Augmented Generation (RAG) chatbot powered by FastAPI, LangGraph, Google Gemini, and Pinecone.

---

## 🏗️ Architecture

The application is structured into three main layers:

1. **Ingestion & Embedding Layer (`app/rag.py`)**:
   * Extracts text from local PDF files in the `data/` directory using `PyPDFLoader`.
   * Splits text into semantically coherent blocks using `RecursiveCharacterTextSplitter`.
   * Generates vector embeddings via `GoogleGenerativeAIEmbeddings` (`models/gemini-embedding-2`) and stores them inside a Pinecone serverless index using AWS `us-east-1` region.
2. **LangGraph Orchestration (`app/rag.py`)**:
   * Uses a StateGraph to coordinate the retrieval and generation workflow:
     * **`retrieve` Node**: Queries Pinecone, extracts the top 12 matches, performs client-side relevance boosting (entity matching and comparison boosts), filters duplicate chunks via Jaccard similarity, and returns the top 3 chunks.
     * **`generate` Node**: Feeds retrieved chunks as context to `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) using strict system instructions to guarantee grounded answers.
3. **Application Interfaces**:
   * **FastAPI Backend (`app/main.py`)**: Hosts a `POST /chat` endpoint returning validated JSON containing the answer, similarity scores, and a confidence rating.
   * **Streamlit UI (`streamlit_app.py`)**: A chat input interface displaying grounded answers, colored status indicators (success, info, warning) based on the confidence score, and expandable context blocks.

---

## ⚙️ Setup Instructions

### 1. Initialize Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and fill in your actual API keys:
```env
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 4. Ingest PDF Document
Place your target PDF inside the `data/` directory and rename it to `agentic_ai.pdf`.

### 5. Start the Services
Start the FastAPI backend:
```bash
uvicorn app.main:app --reload --port 8001
```

In a new terminal window, start the Streamlit client interface:
```bash
python -m streamlit run streamlit_app.py
```

---

## 💬 Sample Queries

Test the chatbot using these sample questions:
1. `What is Agentic AI?`
2. `Compare Non-agentic AI and Agentic AI`
3. `Compare LLMs and Agentic AI`
4. `Compare RPA and Agentic AI`
5. `What are the defining characteristics of an agent?`
6. `What are foundational, workflow, and utility agents?`
