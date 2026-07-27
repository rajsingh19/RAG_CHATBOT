import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import compile_rag_graph

# Initialize FastAPI application
app = FastAPI(title="RAG Chatbot API")

# Define target Pinecone index name
INDEX_NAME = "rag-chatbot-index"

# Compile the LangGraph RAG workflow globally
graph_app = compile_rag_graph(INDEX_NAME)

# Pydantic schema for POST request input validation
class ChatRequest(BaseModel):
    question: str

# Pydantic schema for response structure matching requested JSON
class ChatResponse(BaseModel):
    answer: str
    context: list[str]
    scores: list[float]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    HTTP POST Endpoint that accepts a question, runs the LangGraph RAG pipeline,
    and returns the grounded LLM answer along with retrieved text context chunks
    and their similarity scores from Pinecone.
    """
    # Reject empty questions
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        # Run the LangGraph workflow synchronously with the query state
        result = graph_app.invoke({"question": request.question.strip()})
        
        # Build and return the validated response schema
        return ChatResponse(
            answer=result.get("answer", "No answer generated."),
            context=result.get("retrieved_chunks", []),
            scores=result.get("retrieved_scores", [])
        )
    except Exception as e:
        # Catch unexpected pipeline exceptions and return them as standard server errors
        raise HTTPException(status_code=500, detail=str(e))
