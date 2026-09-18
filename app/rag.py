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
    claims: list
    claim_support_ratio: float
    unsupported_numeric_claims: int
    contradiction_count: int
    query_relevant: bool
    relevance_score: float
    relevance_reason: str
    missing_aspects: list
    requirements: list
    answer_status: str


def evaluate_relevance(query: str, answer: str) -> Dict[str, Any]:
    """
    Independent Query-Answer Relevance Evaluator.
    Input ONLY:
      - query
      - answer

    Evaluates whether the generated answer actually addresses the user's intent.
    Checks:
      1. Specific topic / entity match (e.g. Asking for Perception does not accept Planning).
      2. Comparison completeness (e.g. Compare X and Y requires comparing both entities).
      3. Multi-part requirements (e.g. List X AND explain functions requires both).
      4. Numerical queries (e.g. What percentage requires the specific metric).
      5. Valid short answers are NOT penalized.
      6. Explicit abstentions on unanswerable queries are recognized.
    """
    ans_lower = answer.lower().strip()
    is_abstention = "could not find this information in the provided document" in ans_lower

    if is_abstention:
        return {
            "query_relevant": True,
            "relevance_score": 1.0,
            "reason": "Answer is an explicit abstention stating document information absence.",
            "missing_aspects": [],
            "requirements": [{"requirement": "State absence of document evidence", "addressed": True}]
        }

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)

    system_prompt = (
        "You are an objective Query-Answer Relevance Evaluator.\n"
        "Your task is to determine whether the generated answer actually addresses what the user is asking.\n\n"
        "Guidelines:\n"
        "1. Identify the core intent and requirements of the query:\n"
        "   - Factual definition (e.g. \"What does X do?\") -> Must explain X, not a different entity.\n"
        "   - Comparison (e.g. \"difference between X and Y\", \"compare X and Y\") -> Must address BOTH entities.\n"
        "     If it only explains one entity, relevance is partial (relevance_score ~ 0.30 - 0.50, query_relevant = false).\n"
        "   - Multi-part requirements (e.g. \"What are the six pillars AND what does each do?\") -> Must address all parts.\n"
        "     If it lists pillars without explaining their functions, relevance is partial (relevance_score ~ 0.50, query_relevant = false).\n"
        "   - Numerical query (e.g. \"What percentage...?\") -> Must provide or address the specific metric/figure.\n"
        "2. Do NOT penalize valid short or concise answers if they directly answer the prompt.\n"
        "3. If the answer describes an entirely different topic or entity than requested, query_relevant = false and relevance_score = 0.0.\n"
        "4. Output JSON ONLY matching this schema:\n"
        "{\n"
        "  \"query_relevant\": boolean,\n"
        "  \"relevance_score\": float,\n"
        "  \"reason\": \"string\",\n"
        "  \"missing_aspects\": [\"string\"],\n"
        "  \"requirements\": [\n"
        "    {\"requirement\": \"string\", \"addressed\": boolean}\n"
        "  ]\n"
        "}"
    )

    user_prompt = f"User Query: {query}\n\nGenerated Answer: {answer}\n\nEvaluate and return JSON:"

    try:
        res = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        content = res.content
        if isinstance(content, list):
            content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in relevance output.")

        data = json.loads(json_match.group(0))
        rel_score = float(data.get("relevance_score", 0.0))
        reqs = data.get("requirements", [])
        all_reqs_addressed = all(r.get("addressed", False) for r in reqs) if reqs else (rel_score >= 0.75)
        is_relevant = (rel_score >= 0.75) and all_reqs_addressed

        return {
            "query_relevant": is_relevant,
            "relevance_score": round(rel_score, 4),
            "reason": data.get("reason", ""),
            "missing_aspects": data.get("missing_aspects", []),
            "requirements": reqs
        }
    except Exception as e:
        print(f"Relevance evaluation fallback due to error: {e}")
        return {
            "query_relevant": True,
            "relevance_score": 1.0,
            "reason": f"Fallback: {e}",
            "missing_aspects": [],
            "requirements": []
        }


def compute_lexical_fallback(query: str, answer: str, context: str) -> Dict[str, Any]:
    """Lightweight secondary fallback evaluator using token overlap and regex numeric verification."""
    ans_lower = answer.lower().strip()
    ctx_lower = context.lower()
    q_lower = query.lower()

    # Deterministic regex for numerical claims
    ans_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", answer))
    ctx_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", context))
    real_ans_nums = {n for n in ans_nums if not re.match(r"^[1-9]\.?$", n)}
    unsupported_nums = [n for n in real_ans_nums if n not in ctx_nums]

    stopwords = {
        "what", "are", "the", "of", "an", "and", "does", "each", "do", "in", "is",
        "for", "to", "from", "how", "why", "a", "this", "that", "it", "on", "with",
        "as", "by", "or", "tell", "me", "about", "describe", "explain"
    }

    sentences = [s.strip() for s in re.split(r"[\n\.]+", answer) if len(s.strip()) > 15]
    substantive_sentences = [
        s for s in sentences 
        if "based on the provided document" not in s.lower() and "as follows" not in s.lower()
    ]

    claims = []
    supported_count = 0
    for s in substantive_sentences:
        s_words = [w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", s.lower()) if w not in stopwords]
        if not s_words:
            continue
        found = sum(1 for w in s_words if w in ctx_lower)
        is_sup = (found / len(s_words)) >= 0.65
        if is_sup:
            supported_count += 1
        claims.append({
            "claim": s,
            "supported": is_sup,
            "evidence": "Lexical overlap match" if is_sup else "",
            "contradicted": False
        })

    total_sub = max(1, len(substantive_sentences))
    claim_ratio = round(supported_count / total_sub, 4)
    unsup_num_count = len(unsupported_nums)
    is_grounded = (claim_ratio >= 0.75) and (unsup_num_count == 0)

    if is_grounded and claim_ratio >= 0.85:
        confidence = "High"
    elif is_grounded:
        confidence = "Medium"
    elif claim_ratio >= 0.40:
        confidence = "Low"
    else:
        confidence = "Not Grounded"

    return {
        "claims": claims,
        "claim_support_ratio": claim_ratio,
        "contradiction_count": 0,
        "unsupported_numeric_claims": unsup_num_count,
        "grounded": is_grounded,
        "confidence": confidence,
        "evaluation_source": "lexical_fallback"
    }


def evaluate_grounding(query: str, answer: str, context: str) -> Dict[str, Any]:
    """
    Independent Grounding Evaluator.
    Receives ONLY:
      - query
      - answer
      - context

    Evaluates:
      1. Explicit detection of unsupported numerical claims (via regex + LLM)
      2. Detection of contradictions between answer claims and retrieved context
      3. Support for legitimate paraphrases instead of strict lexical matching
      4. Structured evidence evaluation output:
         {
           "claims": [{"claim": "...", "supported": bool, "evidence": "...", "contradicted": bool}],
           "claim_support_ratio": float,
           "contradiction_count": int,
           "unsupported_numeric_claims": int,
           "grounded": bool,
           "confidence": str
         }
    """
    ans_lower = answer.lower().strip()
    if "could not find this information in the provided document" in ans_lower or not context.strip():
        return {
            "claims": [],
            "claim_support_ratio": 0.0,
            "contradiction_count": 0,
            "unsupported_numeric_claims": 0,
            "grounded": False,
            "confidence": "Not Grounded",
            "evaluation_source": "abstention_rule"
        }

    # 1. Deterministic numerical verification
    ans_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", answer))
    ctx_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", context))
    real_ans_nums = {n for n in ans_nums if not re.match(r"^[1-9]\.?$", n)}
    unsupported_nums = [n for n in real_ans_nums if n not in ctx_nums]

    # 2. Independent LLM evaluator with structured schema
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)

    system_prompt = (
        "You are an objective grounding evaluator. Your job is to verify if the claims in the generated answer "
        "are strictly supported by the provided document context.\n\n"
        "Instructions:\n"
        "1. Extract the atomic factual claims made in the answer.\n"
        "2. For each claim, evaluate:\n"
        "   - \"claim\": concise claim text\n"
        "   - \"supported\": true if directly supported or faithfully paraphrased by the context; false if unsupported or outside context\n"
        "   - \"evidence\": quote or reference from context supporting the claim (empty string if unsupported)\n"
        "   - \"contradicted\": true if directly contradicted by context; false otherwise\n"
        "3. Check for any unsupported numerical claims (numbers, metrics, percentages not substantiated by context).\n"
        "4. Output JSON ONLY matching this structure:\n"
        "{\n"
        "  \"claims\": [\n"
        "    {\"claim\": \"...\", \"supported\": true, \"evidence\": \"...\", \"contradicted\": false}\n"
        "  ],\n"
        "  \"unsupported_numeric_claims\": 0,\n"
        "  \"contradiction_count\": 0\n"
        "}"
    )

    user_prompt = (
        f"User Query: {query}\n\n"
        f"Retrieved Document Context:\n{context}\n\n"
        f"Generated Answer:\n{answer}\n\n"
        f"Evaluate and return JSON:"
    )

    try:
        res = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        content = res.content
        if isinstance(content, list):
            content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
        
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON structure found in evaluator output.")

        data = json.loads(json_match.group(0))
        claims = data.get("claims", [])
        contradiction_count = int(data.get("contradiction_count", 0))
        unsupported_numeric = max(len(unsupported_nums), int(data.get("unsupported_numeric_claims", 0)))

        # Also tally contradictions directly from claim list
        claims_contradicted = sum(1 for c in claims if c.get("contradicted"))
        contradiction_count = max(contradiction_count, claims_contradicted)

        supported_claims = sum(1 for c in claims if c.get("supported") and not c.get("contradicted"))
        total_claims = max(1, len(claims))
        claim_support_ratio = round(supported_claims / total_claims, 4)

        # Grounding decision
        is_grounded = (claim_support_ratio >= 0.75) and (contradiction_count == 0) and (unsupported_numeric == 0)

        # Confidence decision
        if is_grounded and claim_support_ratio >= 0.85:
            confidence = "High"
        elif is_grounded:
            confidence = "Medium"
        elif claim_support_ratio >= 0.40:
            confidence = "Low"
        else:
            confidence = "Not Grounded"

        return {
            "claims": claims,
            "claim_support_ratio": claim_support_ratio,
            "contradiction_count": contradiction_count,
            "unsupported_numeric_claims": unsupported_numeric,
            "grounded": is_grounded,
            "confidence": confidence,
            "evaluation_source": "llm_evaluator"
        }
    except Exception as e:
        print(f"Independent LLM evaluator encountered error ({e}); using lexical fallback.")
        return compute_lexical_fallback(query, answer, context)


def compile_rag_graph(index_name: str = INDEX_NAME) -> StateGraph:
    """Compiles the LangGraph RAG workflow with hybrid retrieval, generation, and independent grounding evaluation."""
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

        return {"answer": ans}

    def evaluate_grounding_node(state: GraphState) -> dict:
        query = state["question"]
        answer = state["answer"]
        context = state["context"]

        # Independent grounding evaluation receiving ONLY query, answer, context
        eval_result = evaluate_grounding(query, answer, context)

        return {
            "answer_grounded": eval_result["grounded"],
            "answer_confidence": eval_result["confidence"],
            "claims": eval_result.get("claims", []),
            "claim_support_ratio": eval_result.get("claim_support_ratio", 0.0),
            "unsupported_numeric_claims": eval_result.get("unsupported_numeric_claims", 0),
            "contradiction_count": eval_result.get("contradiction_count", 0)
        }

    def evaluate_relevance_node(state: GraphState) -> dict:
        query = state["question"]
        answer = state["answer"]

        # Independent query-answer relevance evaluation receiving ONLY query, answer
        rel_result = evaluate_relevance(query, answer)
        is_relevant = rel_result["query_relevant"]
        relevance_score = rel_result["relevance_score"]
        is_grounded = state.get("answer_grounded", True)
        is_abstention = "could not find this information in the provided document" in answer.lower()

        # Conceptual answer status determination
        if is_abstention:
            answer_status = "Legitimate Abstention"
        elif is_grounded and is_relevant:
            answer_status = "Grounded & Relevant"
        elif is_grounded and not is_relevant:
            answer_status = "Generation Failure (Irrelevant to Query)"
        elif not is_grounded:
            answer_status = "Not Grounded (Hallucination / Unsupported)"
        elif relevance_score < 0.75:
            answer_status = "Partially Relevant"
        else:
            answer_status = "Grounded & Relevant"

        return {
            "query_relevant": is_relevant,
            "relevance_score": relevance_score,
            "relevance_reason": rel_result.get("reason", ""),
            "missing_aspects": rel_result.get("missing_aspects", []),
            "requirements": rel_result.get("requirements", []),
            "answer_status": answer_status
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate_grounding", evaluate_grounding_node)
    workflow.add_node("evaluate_relevance", evaluate_relevance_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate_grounding")
    workflow.add_edge("evaluate_grounding", "evaluate_relevance")
    workflow.add_edge("evaluate_relevance", END)

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
