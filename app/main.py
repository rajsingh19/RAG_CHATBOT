import os
import time
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import compile_rag_graph, log_stage_timing

# Initialize FastAPI application
app = FastAPI(title="RAG Chatbot API")

# Define target Pinecone index name
INDEX_NAME = "rag-chatbot-index"

# Compile the LangGraph RAG workflow globally
graph_app = compile_rag_graph(INDEX_NAME)

# Pydantic schema for POST request input validation
class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []

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
    request_id: str = ""
    answer: str
    context: list[ContextChunk]
    confidence: float  # Maintained for backward compatibility
    retrieval_score: float = 0.0
    answer_grounded: bool = True
    answer_confidence: str = "High"
    claim_support_ratio: float = 1.0
    unsupported_numeric_claims: int = 0
    contradiction_count: int = 0
    claims: list[dict] = []
    query_relevant: bool = True
    relevance_score: float = 1.0
    relevance_reason: str = ""
    missing_aspects: list[str] = []
    requirements: list[dict] = []
    answer_status: str = "Grounded & Relevant"
    original_query: str = ""
    resolved_query: str = ""
    retrieval_query: str = ""
    resolved_entities: list[str] = []
    document_concepts: list[str] = []
    context_entities: list[str] = []
    retrieval_fallback_triggered: bool = False

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    HTTP POST Endpoint that accepts a question and optional conversation history,
    runs the LangGraph RAG pipeline asynchronously without blocking the event loop,
    and returns the grounded LLM answer, retrieved chunks, confidence metrics, and query rewrite metadata.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start_total = time.perf_counter()

    # Reject empty questions
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Stage 1: Request Received
    log_stage_timing(
        request_id,
        "1. Chat Request Received",
        "SUCCESS",
        0.0,
        api_call="none",
        retries=0
    )

    try:
        # Offload synchronous LangGraph workflow to a worker thread so the FastAPI
        # asyncio event loop thread is NEVER blocked.
        result = await asyncio.to_thread(
            graph_app.invoke,
            {
                "question": request.question.strip(),
                "history": request.history,
                "request_id": request_id
            }
        )

        t_ser_start = time.perf_counter()

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

        response_obj = ChatResponse(
            request_id=request_id,
            answer=result.get("answer", "No answer generated."),
            context=context_chunks,
            confidence=result.get("confidence", 0.0),
            retrieval_score=result.get("retrieval_score", 0.0),
            answer_grounded=result.get("answer_grounded", True),
            answer_confidence=result.get("answer_confidence", "High"),
            claim_support_ratio=result.get("claim_support_ratio", 1.0),
            unsupported_numeric_claims=result.get("unsupported_numeric_claims", 0),
            contradiction_count=result.get("contradiction_count", 0),
            claims=result.get("claims", []),
            query_relevant=result.get("query_relevant", True),
            relevance_score=result.get("relevance_score", 1.0),
            relevance_reason=result.get("relevance_reason", ""),
            missing_aspects=result.get("missing_aspects", []),
            requirements=result.get("requirements", []),
            answer_status=result.get("answer_status", "Grounded & Relevant"),
            original_query=result.get("original_query", request.question.strip()),
            resolved_query=result.get("resolved_query", request.question.strip()),
            retrieval_query=result.get("retrieval_query", request.question.strip()),
            resolved_entities=result.get("resolved_entities", []),
            document_concepts=result.get("document_concepts", []),
            context_entities=result.get("context_entities", []),
            retrieval_fallback_triggered=result.get("retrieval_fallback_triggered", False)
        )

        elapsed_ser_ms = (time.perf_counter() - t_ser_start) * 1000
        total_elapsed_ms = (time.perf_counter() - t_start_total) * 1000

        # Stage 7: Final Response Serialization
        log_stage_timing(
            request_id,
            "7. Response Serialization",
            "SUCCESS",
            elapsed_ser_ms,
            api_call="none",
            retries=0
        )
        log_stage_timing(
            request_id,
            "Total Pipeline Roundtrip",
            "SUCCESS",
            total_elapsed_ms,
            api_call="all",
            retries=0
        )

        return response_obj

    except Exception as e:
        total_elapsed_ms = (time.perf_counter() - t_start_total) * 1000
        log_stage_timing(
            request_id,
            "Pipeline Execution",
            "FAILURE",
            total_elapsed_ms,
            api_call="pipeline",
            retries=0,
            exception=str(e)
        )
        # Wrap and return 500 error on internal pipeline failures
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

