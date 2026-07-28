import os
import time
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from langgraph.graph import StateGraph, START, END

# 1. Load environment variables from the .env file
load_dotenv()

# 2. Initialize global API keys and clients
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone client once globally
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None

# Initialize Gemini Embeddings model globally (reused for both ingestion and retrieval)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2") if GOOGLE_API_KEY else None


def process_pdf_and_embed(file_path: str):
    """
    Loads a PDF, splits its content into chunks, and generates embeddings for ALL chunks.
    Uses rate-limit-safe batching (50 chunks per batch) to avoid Gemini Free Tier 429 errors.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'.")
        return None, None

    # Load PDF using PyPDFLoader
    print(f"Loading PDF from: {file_path}...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Successfully loaded {len(pages)} pages.")

    # Split PDF text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    print(f"Successfully split into {len(chunks)} chunks.")

    if not chunks:
        return None, None

    # Generate Embeddings in rate-limit-safe batches of 50
    print("Generating embeddings for all chunks...")
    vector_list = []
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [chunk.page_content for chunk in batch_chunks]
        
        print(f"Embedding chunks {i + 1} to {min(i + batch_size, len(chunks))} of {len(chunks)}...")
        batch_vectors = embeddings.embed_documents(batch_texts)
        vector_list.extend(batch_vectors)
        
        # 15s sleep to stay under Gemini free-tier rate limits (100 requests per minute)
        if i + batch_size < len(chunks):
            print("Sleeping for 15 seconds to avoid Gemini rate limits...")
            time.sleep(15)

    print(f"Successfully generated {len(vector_list)} embeddings.")
    return chunks, vector_list


def setup_pinecone_index(index_name: str, dimension: int):
    """
    Checks if Pinecone index exists, and creates a serverless index if it doesn't.
    """
    if not pc:
        print("Error: Pinecone client not initialized. Check PINECONE_API_KEY.")
        return None

    print(f"Connecting to Pinecone and checking index '{index_name}'...")
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' not found. Creating a new one...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Successfully created index '{index_name}'!")
    else:
        print(f"Index '{index_name}' already exists.")
        
    return pc


def upload_vectors_to_pinecone(index_name: str, chunks: list, vector_list: list):
    """
    Uploads all chunk embeddings to Pinecone, storing the text content in metadata.
    """
    if not pc:
        print("Error: Pinecone client not initialized.")
        return

    index = pc.Index(index_name)
    vectors_to_upsert = []
    
    for idx, (chunk, vector) in enumerate(zip(chunks, vector_list)):
        vectors_to_upsert.append((
            str(idx),  # Simple numeric ID
            vector,    # Float embedding values
            {"text": chunk.page_content, "page": chunk.metadata.get("page", 0)}  # Metadata
        ))
        
    print(f"Uploading {len(vectors_to_upsert)} vectors to index '{index_name}'...")
    
    # Upsert in batches of 50 to log progress
    upsert_batch_size = 50
    for i in range(0, len(vectors_to_upsert), upsert_batch_size):
        batch = vectors_to_upsert[i : i + upsert_batch_size]
        index.upsert(vectors=batch)
        print(f"Uploaded vectors {i + 1} to {min(i + upsert_batch_size, len(vectors_to_upsert))}...")
        
    print("Upload complete!")


# Define the state variables shared across our LangGraph nodes
class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    retrieved_chunks: list  # List of dicts: [{"text": str, "score": float}]
    confidence: float       # Highest similarity score


def compile_rag_graph(index_name: str) -> StateGraph:
    """
    Builds and compiles the RAG StateGraph workflow.
    Graph flow: START -> retrieve -> generate -> END
    """
    if not pc:
        raise ValueError("Pinecone client not initialized.")

    # Node 1: Retrieve matching chunks from Pinecone
    def retrieve_node(state: GraphState) -> dict:
        query = state["question"]
        print("-> [Node: Retrieve] Searching Pinecone vector index...")
        
        index = pc.Index(index_name)
        query_vector = embeddings.embed_query(query)
        
        # 1. Query more results (top 12) from Pinecone to perform client-side re-ranking
        results = index.query(vector=query_vector, top_k=12, include_metadata=True)
        matches = results.get("matches", [])
        
        # 2. Parse keywords from query (ignore casing and typical question/comparison stop words)
        stop_words = {"compare", "versus", "vs", "difference", "differences", "between", "and", "what", "is", "are", "the", "a", "an", "of", "in", "to", "for", "with", "similarities"}
        query_words = [w.strip("?,.!:;").lower() for w in query.split()]
        keywords = [w for w in query_words if w not in stop_words and len(w) > 2]
        
        is_comparison_query = any(w in query_words for w in ["compare", "versus", "vs", "difference", "differences", "comparison", "contrast", "between"])
        
        # 3. Score each chunk based on relevance boosts
        scored_matches = []
        for match in matches:
            text = match.get("metadata", {}).get("text", "")
            if not text:
                continue
            
            score = match.get("score", 0.0)
            text_lower = text.lower()
            
            # Boost score if the chunk contains query keywords (Co-occurrence/Entity matching)
            matched_keywords = sum(1 for kw in keywords if kw in text_lower)
            if keywords:
                keyword_ratio = matched_keywords / len(keywords)
                score += keyword_ratio * 0.35  # Up to 0.35 boost for entity co-occurrence
                
            # Boost score if this is a comparison query and the chunk contains comparison cues
            if is_comparison_query:
                comparison_indicators = ["vs", "versus", "compare", "comparison", "difference", "differ", "table", "contrast", "on the other hand", "whereas", "while"]
                found_indicators = sum(1 for ci in comparison_indicators if ci in text_lower)
                if found_indicators > 0:
                    score += 0.25  # Up to 0.25 boost for matching comparison concepts
            
            scored_matches.append((match, score))
            
        # Sort matched chunks by their newly calculated relevance score (descending)
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Filter for Diversity (Jaccard similarity to prune redundant/duplicate chunks)
        selected_matches = []
        
        def get_words_set(t):
            return set(t.lower().split())
            
        for match, boosted_score in scored_matches:
            if len(selected_matches) >= 3:
                break
                
            text = match.get("metadata", {}).get("text", "")
            words_set = get_words_set(text)
            
            is_duplicate = False
            for sel_match in selected_matches:
                sel_text = sel_match.get("metadata", {}).get("text", "")
                sel_words = get_words_set(sel_text)
                
                # Jaccard index calculation
                intersection = len(words_set.intersection(sel_words))
                union = len(words_set.union(sel_words))
                jaccard = intersection / union if union > 0 else 0.0
                
                # Skip if word overlap is greater than 50%
                if jaccard > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected_matches.append(match)
                
        # Fill remaining slots up to 3 if diversity filter was too strict
        if len(selected_matches) < 3:
            for match, boosted_score in scored_matches:
                if len(selected_matches) >= 3:
                    break
                if match not in selected_matches:
                    selected_matches.append(match)
                    
        # 5. Format final top 3 chunks (prepend Page Number to display text)
        retrieved_chunks = []
        scores = []
        for match in selected_matches[:3]:
            text = match.get("metadata", {}).get("text", "")
            score = match.get("score", 0.0)
            page = match.get("metadata", {}).get("page", None)
            
            # Embed page metadata directly into text for UI display purposes
            display_text = f"[Page {page + 1}] {text}" if page is not None else text
            
            retrieved_chunks.append({"text": display_text, "score": score})
            scores.append(score)
            
        context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        confidence = scores[0] if scores else 0.0
        
        return {
            "context": context,
            "retrieved_chunks": retrieved_chunks,
            "confidence": confidence
        }

    # Node 2: Generate response using LLM grounded in context
    def generate_node(state: GraphState) -> dict:
        query = state["question"]
        context = state["context"]
        print("-> [Node: Generate] Grounding and generating answer with Gemini LLM...")
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        
        system_instruction = (
            "You are a helpful assistant. You must answer the user's question ONLY using the provided context. "
            "Do not add any external knowledge or make assumptions. "
            "If the answer is not explicitly present in the provided context, you must reply exactly with: "
            "\"I could not find this information in the provided document.\"\n"
            "Return only the answer."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
        
        response = llm.invoke([
            ("system", system_instruction),
            ("human", user_prompt)
        ])
        return {"answer": response.content}

    # Connect nodes into the StateGraph
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


def query_rag_with_graph(query: str, index_name: str):
    """
    Invokes the compiled StateGraph workflow and prints the final grounded answer.
    """
    app = compile_rag_graph(index_name)
    print(f"\nInvoking LangGraph for query: '{query}'...")
    result = app.invoke({"question": query})
    
    print("\n" + "="*40)
    print("ANSWER (via LangGraph):")
    print("="*40)
    print(result.get("answer", "No answer generated."))
    print("="*40)


if __name__ == "__main__":
    index_name = "rag-chatbot-index"
    pdf_path = "data/agentic_ai.pdf"
    
    if not PINECONE_API_KEY:
        print("Error: Please set PINECONE_API_KEY in the .env file.")
        exit(1)
        
    # Check if index exists and contains vectors
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' not found. Running ingestion pipeline...")
        chunks, vector_list = process_pdf_and_embed(pdf_path)
        if chunks and vector_list:
            setup_pinecone_index(index_name, len(vector_list[0]))
            upload_vectors_to_pinecone(index_name, chunks, vector_list)
    else:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            print(f"Index '{index_name}' is empty. Running ingestion pipeline...")
            chunks, vector_list = process_pdf_and_embed(pdf_path)
            if chunks and vector_list:
                upload_vectors_to_pinecone(index_name, chunks, vector_list)
        else:
            print(f"Index '{index_name}' is already populated with {stats['total_vector_count']} vectors.")

    # Start the interactive querying loop
    print("\nStarting RAG Query & Retrieval Testing Interface. Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nEnter search query: ")
            if query.strip().lower() == "exit":
                print("Exiting search interface.")
                break
            if not query.strip():
                continue
            
            query_rag_with_graph(query.strip(), index_name)
            
        except KeyboardInterrupt:
            print("\nExiting search interface.")
            break
