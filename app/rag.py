import os
import re
import json
import time
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
import pypdf
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from langgraph.graph import StateGraph, START, END
from rank_bm25 import BM25Okapi

# 1. Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
DEBUG_RETRIEVAL = os.environ.get("DEBUG_RETRIEVAL", "true").lower() in ("true", "1", "yes")

# 2. Clients
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2") if GOOGLE_API_KEY else None

INDEX_NAME = "rag-chatbot-index"
PDF_PATH = "data/agentic_ai.pdf"
CHUNKS_CACHE_FILE = "data/chunks_cache.json"

# In-memory corpus and BM25 index globals
_CHUNKS_CORPUS: List[Dict[str, Any]] = []
_BM25_INDEX: Optional[BM25Okapi] = None


def tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25 keyword retrieval, lowercasing and stripping punctuation."""
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1]


def parse_and_chunk_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Structure-aware document parser and semantic chunker for the Agentic AI PDF.
    Preserves:
      - document name
      - chapter context
      - section heading
      - physical page index (pdf_page_number)
      - printed page number (printed_page_number)
      - semantic blocks, numbered items, examples, and definitions
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    reader = pypdf.PdfReader(file_path)
    total_pages = len(reader.pages)
    
    chapter_map = {
        (1, 5): "Front Matter & Table of Contents",
        (6, 16): "Chapter 1: Introduction to Agentic AI",
        (17, 28): "Chapter 2: Anatomy of an Agentic AI System",
        (29, 37): "Chapter 3: Multi-Agent Systems",
        (38, 49): "Chapter 4: Orchestrating Agentic AI Systems",
        (50, 54): "Chapter 5: Your Readiness for Agentic AI",
        (55, 60): "Chapter 6: Practical Applications of Agentic AI",
    }

    def get_chapter(page_num: int) -> str:
        for (start, end), ch in chapter_map.items():
            if start <= page_num <= end:
                return ch
        return "General"

    chunks: List[Dict[str, Any]] = []
    current_section = "Overview"

    prev_chapter = None
    for p_idx in range(total_pages):
        pdf_page = p_idx + 1
        page = reader.pages[p_idx]
        raw_text = page.extract_text() or ""
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        if not lines:
            continue

        chapter = get_chapter(pdf_page)
        if chapter != prev_chapter:
            current_section = "Overview"
            prev_chapter = chapter

        # Detect printed page number from lines (usually near top or bottom or standalone digit)
        printed_page = None
        for l in lines:
            if re.match(r"^(?:0[1-9]|[1-5][0-9]|60|[1-9])$", l):
                printed_page = int(l)
                break

        # Detect section header
        for l in lines:
            # Matches e.g. "1.1 The Terminology Maze", "2.1 The Core Pillars: From Perception to Execution"
            if re.match(r"^[1-6]\.[0-9]+(?:\.[0-9]+)?\s+[A-Za-z]", l):
                current_section = l
                break

        # Filter out repeated decorative running headers / footers
        filtered_lines = []
        for l in lines:
            if l == "AGENTIC AI FOR EXECUTIVES":
                continue
            if re.match(r"^(?:0[1-9]|[1-5][0-9]|60|[1-9])$", l):
                continue
            filtered_lines.append(l)

        page_body = "\n".join(filtered_lines).strip()
        if not page_body:
            continue

        # Semantic chunk sizing:
        # Keep coherent sections / pages intact up to ~2300 characters.
        # If longer, split only at semantic paragraph / concept boundaries.
        if len(page_body) <= 2300:
            semantic_blocks = [page_body]
        else:
            paragraphs = page_body.split("\n\n")
            semantic_blocks = []
            cur_p = []
            cur_len = 0
            for p in paragraphs:
                if cur_len + len(p) > 1800 and cur_p:
                    semantic_blocks.append("\n\n".join(cur_p))
                    cur_p = [p]
                    cur_len = len(p)
                else:
                    cur_p.append(p)
                    cur_len += len(p)
            if cur_p:
                semantic_blocks.append("\n\n".join(cur_p))

        for block_idx, block in enumerate(semantic_blocks):
            chunk_id = f"chunk_p{pdf_page}_{block_idx}"
            chunk_metadata = {
                "chunk_id": chunk_id,
                "document": "Agentic AI for Executives",
                "chapter": chapter,
                "section": current_section,
                "pdf_page_number": pdf_page,
                "printed_page_number": printed_page if printed_page is not None else pdf_page,
                "content": block
            }
            
            # Contextual prefix prepended to text before embedding
            contextual_prefix = (
                f"Document: {chunk_metadata['document']}\n"
                f"Chapter: {chunk_metadata['chapter']}\n"
                f"Section: {chunk_metadata['section']}\n"
                f"PDF Page: {chunk_metadata['pdf_page_number']}\n"
                f"Printed Page: {chunk_metadata['printed_page_number']}\n\n"
                f"Content:\n{block}"
            )
            chunk_metadata["contextual_text"] = contextual_prefix
            chunks.append(chunk_metadata)

    return chunks


def load_chunks_corpus(force_reparse: bool = False) -> List[Dict[str, Any]]:
    """Loads chunks from local JSON cache if available, otherwise parses PDF and caches them."""
    global _CHUNKS_CORPUS, _BM25_INDEX
    if _CHUNKS_CORPUS and not force_reparse:
        return _CHUNKS_CORPUS

    if os.path.exists(CHUNKS_CACHE_FILE) and not force_reparse:
        try:
            with open(CHUNKS_CACHE_FILE, "r", encoding="utf-8") as f:
                _CHUNKS_CORPUS = json.load(f)
        except Exception:
            _CHUNKS_CORPUS = []

    if not _CHUNKS_CORPUS:
        _CHUNKS_CORPUS = parse_and_chunk_pdf(PDF_PATH)
        os.makedirs(os.path.dirname(CHUNKS_CACHE_FILE), exist_ok=True)
        with open(CHUNKS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_CHUNKS_CORPUS, f, indent=2)

    # Initialize BM25 index over contextual texts
    tokenized_corpus = [tokenize(c["contextual_text"]) for c in _CHUNKS_CORPUS]
    _BM25_INDEX = BM25Okapi(tokenized_corpus)
    return _CHUNKS_CORPUS


def get_bm25_index() -> tuple[BM25Okapi, List[Dict[str, Any]]]:
    """Ensures BM25 index and chunk corpus are loaded and returns them."""
    global _BM25_INDEX, _CHUNKS_CORPUS
    if _BM25_INDEX is None or not _CHUNKS_CORPUS:
        load_chunks_corpus()
    return _BM25_INDEX, _CHUNKS_CORPUS


def setup_pinecone_index(index_name: str = INDEX_NAME, dimension: int = 3072):
    """Verifies Pinecone index existence, creating a serverless index if needed."""
    if not pc:
        raise ValueError("Pinecone client not initialized.")
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index '{index_name}' (dim={dimension}, metric=cosine)...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)
    return pc.Index(index_name)


def reingest_pdf_to_pinecone(index_name: str = INDEX_NAME, pdf_path: str = PDF_PATH):
    """
    Safely re-ingests the PDF with structure-aware semantic chunks and contextual metadata.
    Clears all existing vectors first to prevent mixing of outdated character chunks.
    """
    if not pc or not embeddings:
        raise ValueError("Pinecone or Gemini Embeddings not configured.")

    index = setup_pinecone_index(index_name, dimension=3072)
    
    print("Parsing PDF with semantic structure-aware chunker...")
    chunks = parse_and_chunk_pdf(pdf_path)
    print(f"Generated {len(chunks)} semantic chunks.")

    # Save to cache
    with open(CHUNKS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    load_chunks_corpus(force_reparse=True)

    print(f"Deleting existing vectors in index '{index_name}' to avoid mixing...")
    try:
        index.delete(delete_all=True)
        time.sleep(2)
    except Exception as e:
        print(f"Note on delete: {e}")

    print("Generating embeddings for all contextual chunks...")
    batch_size = 30
    vectors_to_upsert = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_texts = [c["contextual_text"] for c in batch]
        print(f"Embedding chunks {i + 1} to {min(i + batch_size, len(chunks))} of {len(chunks)}...")
        batch_vectors = embeddings.embed_documents(batch_texts)

        for chunk, vector in zip(batch, batch_vectors):
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "document": chunk["document"],
                "chapter": chunk["chapter"],
                "section": chunk["section"],
                "pdf_page_number": chunk["pdf_page_number"],
                "printed_page_number": chunk["printed_page_number"],
                "text": chunk["content"],
                "contextual_text": chunk["contextual_text"]
            }
            vectors_to_upsert.append((chunk["chunk_id"], vector, metadata))

        if i + batch_size < len(chunks):
            print("Pausing 5 seconds for Gemini API rate limits...")
            time.sleep(5)

    print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
    upsert_batch_size = 50
    for i in range(0, len(vectors_to_upsert), upsert_batch_size):
        batch = vectors_to_upsert[i : i + upsert_batch_size]
        index.upsert(vectors=batch)

    stats = index.describe_index_stats()
    print(f"Ingestion complete! Index stats: {stats}")
    return stats


def hybrid_retrieve(query: str, index_name: str = INDEX_NAME, top_k: int = 15) -> List[Dict[str, Any]]:
    """
    Executes hybrid retrieval:
      1. Dense vector similarity via Pinecone
      2. Keyword / BM25 retrieval via rank_bm25
      3. Reciprocal Rank Fusion (RRF) candidate merging
      4. Multi-chunk section context expansion (prevents partial section truncation)
    """
    if not pc or not embeddings:
        raise ValueError("Pinecone or Gemini Embeddings not configured.")

    index = pc.Index(index_name)
    bm25, corpus = get_bm25_index()
    corpus_by_id = {c["chunk_id"]: c for c in corpus}

    # 1. Dense Pinecone Vector Search
    query_vector = embeddings.embed_query(query)
    dense_results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    dense_matches = dense_results.get("matches", [])

    dense_ranks: Dict[str, int] = {}
    dense_scores: Dict[str, float] = {}
    dense_metadata: Dict[str, Dict[str, Any]] = {}
    for r_idx, match in enumerate(dense_matches):
        c_id = match["id"]
        dense_ranks[c_id] = r_idx + 1
        dense_scores[c_id] = float(match.get("score", 0.0))
        dense_metadata[c_id] = match.get("metadata", {})

    # 2. BM25 Keyword Search
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    bm25_ranks: Dict[str, int] = {}
    bm25_score_map: Dict[str, float] = {}
    for r_idx, idx in enumerate(bm25_ranked_indices):
        c_id = corpus[idx]["chunk_id"]
        bm25_ranks[c_id] = r_idx + 1
        bm25_score_map[c_id] = float(bm25_scores[idx])

    # 3. Reciprocal Rank Fusion (RRF)
    all_candidate_ids = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
    rrf_candidates = []

    for c_id in all_candidate_ids:
        rrf_score = 0.0
        if c_id in dense_ranks:
            rrf_score += 1.0 / (60.0 + dense_ranks[c_id])
        if c_id in bm25_ranks and bm25_score_map.get(c_id, 0.0) > 0:
            rrf_score += 1.0 / (60.0 + bm25_ranks[c_id])

        # Get metadata
        meta = dense_metadata.get(c_id) or corpus_by_id.get(c_id, {})
        d_score = dense_scores.get(c_id, 0.0)
        b_rank = bm25_ranks.get(c_id, None)
        d_rank = dense_ranks.get(c_id, None)

        rrf_candidates.append({
            "chunk_id": c_id,
            "rrf_score": rrf_score,
            "dense_score": d_score,
            "dense_rank": d_rank,
            "bm25_rank": b_rank,
            "metadata": meta
        })

    # Sort candidates by combined RRF score descending
    rrf_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)

    # 4. Multi-Chunk Section Context Expansion:
    # If a top-scoring candidate belongs to a multi-chunk section (e.g. Section 2.1),
    # ensure contiguous / complementary chunks from that section are included.
    top_sections = set()
    for cand in rrf_candidates[:3]:
        sec = cand["metadata"].get("section")
        if sec and sec != "Overview":
            top_sections.add(sec)

    selected_chunks: List[Dict[str, Any]] = []
    selected_ids = set()
    total_char_count = 0
    MAX_CHAR_BUDGET = 7500

    # First pass: high RRF candidates
    for cand in rrf_candidates:
        c_id = cand["chunk_id"]
        text = cand["metadata"].get("text", "")
        if c_id not in selected_ids and total_char_count + len(text) <= MAX_CHAR_BUDGET:
            selected_chunks.append(cand)
            selected_ids.add(c_id)
            total_char_count += len(text)
        if len(selected_chunks) >= 5:
            break

    # Second pass: Check if any chunk from a top section was missed
    for chunk in corpus:
        c_id = chunk["chunk_id"]
        sec = chunk.get("section")
        if sec in top_sections and c_id not in selected_ids:
            text = chunk.get("content", "")
            if total_char_count + len(text) <= MAX_CHAR_BUDGET:
                selected_chunks.append({
                    "chunk_id": c_id,
                    "rrf_score": 0.01,
                    "dense_score": dense_scores.get(c_id, 0.70),
                    "dense_rank": dense_ranks.get(c_id, None),
                    "bm25_rank": bm25_ranks.get(c_id, None),
                    "metadata": chunk
                })
                selected_ids.add(c_id)
                total_char_count += len(text)

    # Format final retrieved chunk items
    final_results = []
    for rank_idx, item in enumerate(selected_chunks):
        meta = item["metadata"]
        score = item["dense_score"] if item["dense_score"] > 0 else 0.75
        final_results.append({
            "chunk_id": item["chunk_id"],
            "rank": rank_idx + 1,
            "score": score,
            "rrf_score": item["rrf_score"],
            "document": meta.get("document", "Agentic AI for Executives"),
            "chapter": meta.get("chapter", ""),
            "section": meta.get("section", ""),
            "pdf_page_number": meta.get("pdf_page_number", 0),
            "printed_page_number": meta.get("printed_page_number", 0),
            "text": meta.get("text", meta.get("content", ""))
        })

    # Debug Mode Logging
    if DEBUG_RETRIEVAL:
        print("\n" + "=" * 30 + " RETRIEVAL DEBUG " + "=" * 30)
        print(f"USER QUERY: {query}")
        print("RETRIEVED CHUNKS:")
        for res in final_results:
            print(f"  Chunk {res['rank']}")
            print(f"    Score: {res['score']:.4f} (RRF: {res['rrf_score']:.5f})")
            print(f"    Document: {res['document']}")
            print(f"    Chapter: {res['chapter']}")
            print(f"    Section: {res['section']}")
            print(f"    PDF Page: {res['pdf_page_number']} | Printed Page: {res['printed_page_number']}")
            print(f"    Chunk ID: {res['chunk_id']}")
            preview = res['text'][:120].replace('\n', ' ')
            print(f"    Text: {preview}...")
        print("=" * 77 + "\n")

    return final_results


# LangGraph State
class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    retrieved_chunks: list
    retrieval_score: float
    confidence: float
    answer_grounded: bool
    answer_confidence: str


def compute_grounding_confidence(
    query: str, 
    answer: str, 
    context: str, 
    retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes answer_grounded and answer_confidence based on actual evidence coverage signals
    rather than mapping raw vector retrieval similarity to confidence.

    Evidence coverage signals:
    1. Abstention detection: If model states it could not find the information, marked Not Grounded.
    2. Answer claim verification (Faithfulness): Proportion of substantive answer sentences whose key terms
       are corroborated by the retrieved context.
    3. Query concept coverage (Completeness): Proportion of content words in the user query found in the retrieved context.
    4. Evidence presence: Number of relevant retrieved context chunks.
    """
    ans_lower = answer.lower().strip()
    ctx_lower = context.lower()
    q_lower = query.lower()

    if "could not find this information in the provided document" in ans_lower or not retrieved_chunks:
        return {
            "answer_grounded": False,
            "answer_confidence": "Not Grounded",
            "claim_support_ratio": 0.0,
            "query_concept_coverage": 0.0,
            "evidence_score": 0.0
        }

    stopwords = {
        "what", "are", "the", "of", "an", "and", "does", "each", "do", "in", "is",
        "for", "to", "from", "how", "why", "a", "this", "that", "it", "on", "with",
        "as", "by", "or", "tell", "me", "about", "describe", "explain"
    }

    # 1. Query Concept Coverage Signal
    q_words = [w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", q_lower) if w not in stopwords]
    q_covered = [w for w in q_words if w in ctx_lower]
    query_concept_coverage = len(q_covered) / len(q_words) if q_words else 1.0

    # 2. Answer Claim Grounding Signal
    sentences = [s.strip() for s in re.split(r"[\n\.]+", answer) if len(s.strip()) > 15]
    substantive_sentences = [
        s for s in sentences 
        if "based on the provided document" not in s.lower() and "as follows" not in s.lower()
    ]

    supported_count = 0
    for s in substantive_sentences:
        s_words = [w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", s.lower()) if w not in stopwords]
        if not s_words:
            continue
        found = sum(1 for w in s_words if w in ctx_lower)
        if (found / len(s_words)) >= 0.65:
            supported_count += 1

    total_substantive = max(1, len(substantive_sentences))
    claim_support_ratio = supported_count / total_substantive

    # 3. Composite Evidence Score
    evidence_score = 0.60 * claim_support_ratio + 0.40 * query_concept_coverage

    # 4. Confidence level determination based on evidence coverage
    if claim_support_ratio >= 0.80 and evidence_score >= 0.75:
        confidence = "High"
        grounded = True
    elif claim_support_ratio >= 0.60 and evidence_score >= 0.55:
        confidence = "Medium"
        grounded = True
    elif claim_support_ratio >= 0.40:
        confidence = "Low"
        grounded = True
    else:
        confidence = "Low"
        grounded = False

    return {
        "answer_grounded": grounded,
        "answer_confidence": confidence,
        "claim_support_ratio": round(claim_support_ratio, 4),
        "query_concept_coverage": round(query_concept_coverage, 4),
        "evidence_score": round(evidence_score, 4),
        "supported_sentences": supported_count,
        "total_substantive_sentences": total_substantive
    }


def compile_rag_graph(index_name: str = INDEX_NAME) -> StateGraph:
    """Compiles the LangGraph RAG workflow with hybrid retrieval and strict grounding."""
    # Ensure in-memory BM25 index is loaded
    load_chunks_corpus()

    def retrieve_node(state: GraphState) -> dict:
        query = state["question"]
        results = hybrid_retrieve(query, index_name=index_name, top_k=6)

        context_parts = []
        scores = []
        formatted_chunks = []

        for r in results:
            header = (
                f"[Section: {r['section']} | "
                f"PDF Page: {r['pdf_page_number']} | "
                f"Printed Page: {r['printed_page_number']}]"
            )
            context_parts.append(f"{header}\n{r['text']}")
            scores.append(r["score"])
            formatted_chunks.append({
                "chunk_id": r["chunk_id"],
                "text": f"{header}\n{r['text']}",
                "score": r["score"],
                "document": r["document"],
                "chapter": r["chapter"],
                "section": r["section"],
                "pdf_page_number": r["pdf_page_number"],
                "printed_page_number": r["printed_page_number"]
            })

        context = "\n\n---\n\n".join(context_parts)
        top_score = max(scores) if scores else 0.0

        return {
            "context": context,
            "retrieved_chunks": formatted_chunks,
            "retrieval_score": top_score,
            "confidence": top_score
        }

    def generate_node(state: GraphState) -> dict:
        query = state["question"]
        context = state["context"]
        retrieved_chunks = state.get("retrieved_chunks", [])

        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", max_retries=6)

        system_instruction = (
            "You are a helpful, precise assistant answering questions about the Agentic AI document.\n"
            "You must answer the user's question ONLY using the provided context.\n\n"
            "Rules:\n"
            "1. Base your answer strictly on the facts and definitions present in the context. Do not use outside knowledge or introduce facts not in the context.\n"
            "2. When the user asks for components, pillars, definitions, or comparisons, explain each concept that is detailed in the context thoroughly.\n"
            "3. If the provided context does not contain sufficient information to answer the question, reply exactly with: "
            "\"I could not find this information in the provided document.\"\n"
            "4. Do not invent Python code or recommend external models if they are not explicitly mentioned in the context."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        # Safe invocation with backoff retry for rate limits
        response = None
        for attempt in range(5):
            try:
                response = llm.invoke([
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ])
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str and attempt < 4:
                    wait_time = 12 * (attempt + 1)
                    print(f"Rate limited (429). Waiting {wait_time}s for quota window to reset (attempt {attempt + 1}/5)...")
                    time.sleep(wait_time)
                else:
                    raise e

        if isinstance(response.content, list):
            ans = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in response.content]).strip()
        else:
            ans = str(response.content).strip()

        # Evidence-coverage-based grounding and confidence calculation
        eval_metrics = compute_grounding_confidence(query, ans, context, retrieved_chunks)

        return {
            "answer": ans,
            "answer_grounded": eval_metrics["answer_grounded"],
            "answer_confidence": eval_metrics["answer_confidence"]
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--reingest":
        print("Executing re-ingestion...")
        reingest_pdf_to_pinecone(INDEX_NAME, PDF_PATH)
    else:
        print("Testing RAG pipeline. To re-ingest run: python -m app.rag --reingest")
        graph = compile_rag_graph(INDEX_NAME)
        q = "What are the six core pillars of an Agentic AI system, and what does each pillar do?"
        res = graph.invoke({"question": q})
        print("\nAnswer:\n", res["answer"])
