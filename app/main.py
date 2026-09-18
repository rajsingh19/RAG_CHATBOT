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

# Represents each retrieved chunk with text, similarity score, and structure metadata
class ContextChunk(BaseModel):
    text: str
    score: float
    document: str = "Agentic AI for Executives"
    chapter: str = ""
    section: str = ""
    pdf_page_number: int = 0
    printed_page_number: int = 0
    chunk_id: str = ""

# Pydantic schema for response structure matching requested JSON
class ChatResponse(BaseModel):
    answer: str
    context: list[ContextChunk]
    confidence: float  # Maintained for backward compatibility
    retrieval_score: float = 0.0
    answer_grounded: bool = True
    answer_confidence: str = "High"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    HTTP POST Endpoint that accepts a question, runs the LangGraph RAG pipeline,
    and returns the grounded LLM answer, the text chunks bundled with their individual
    similarity scores and structural metadata, and separated retrieval vs grounding confidence metrics.
    """
    # Reject empty questions
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        # Run the LangGraph workflow synchronously with the query state
        result = graph_app.invoke({"question": request.question.strip()})
        
        # Format the context chunks into ContextChunk model schemas
        context_chunks = [
            ContextChunk(
                text=chunk.get("text", ""),
                score=chunk.get("score", 0.0),
                document=chunk.get("document", "Agentic AI for Executives"),
                chapter=chunk.get("chapter", ""),
                section=chunk.get("section", ""),
                pdf_page_number=chunk.get("pdf_page_number", 0),
                printed_page_number=chunk.get("printed_page_number", 0),
                chunk_id=chunk.get("chunk_id", "")
            )
            for chunk in result.get("retrieved_chunks", [])
        ]
        
        # Build and return the validated response schema
        return ChatResponse(
            answer=result.get("answer", "No answer generated."),
            context=context_chunks,
            confidence=result.get("confidence", 0.0),
            retrieval_score=result.get("retrieval_score", 0.0),
            answer_grounded=result.get("answer_grounded", True),
            answer_confidence=result.get("answer_confidence", "High")
        )
    except Exception as e:
        # Catch unexpected pipeline exceptions and return them as standard server errors
        raise HTTPException(status_code=500, detail=str(e))
