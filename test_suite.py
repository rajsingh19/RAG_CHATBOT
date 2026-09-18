import json
import time
from typing import Dict, Any, List
from app.rag import compile_rag_graph, hybrid_retrieve, INDEX_NAME

# Initialize graph
graph = compile_rag_graph(INDEX_NAME)

TEST_CASES = [
    {
        "id": "TEST 1",
        "query": "What are the six core pillars of Agentic AI?",
        "expected_section": "2.1 The Core Pillars: From Perception to Execution",
        "expected_pages": [19],
        "expected_concepts": ["Perception", "Reasoning", "Planning", "Learning", "Verification", "Execution"],
        "is_abstention_case": False
    },
    {
        "id": "ORIGINAL FAILING QUERY",
        "query": "What are the six core pillars of an Agentic AI system, and what does each pillar do?",
        "expected_section": "2.1 The Core Pillars: From Perception to Execution",
        "expected_pages": [19],
        "expected_concepts": ["Perception", "Reasoning", "Planning", "Learning", "Verification", "Execution"],
        "is_abstention_case": False
    },
    {
        "id": "PHRASING INVARIANCE 1",
        "query": "Explain the core pillars of Agentic AI.",
        "expected_section": "2.1 The Core Pillars: From Perception to Execution",
        "expected_pages": [19],
        "expected_concepts": ["Perception", "Reasoning", "Planning", "Learning", "Verification", "Execution"],
        "is_abstention_case": False
    },
    {
        "id": "PHRASING INVARIANCE 2",
        "query": "What are the six components involved in the production-line AI example?",
        "expected_section": "2.1 The Core Pillars: From Perception to Execution",
        "expected_pages": [19],
        "expected_concepts": ["Perception", "Reasoning", "Planning", "Learning", "Verification", "Execution"],
        "is_abstention_case": False
    },
    {
        "id": "TEST 2",
        "query": "What does perception mean in an Agentic AI system?",
        "expected_section": "2.1 The Core Pillars: From Perception to Execution",
        "expected_pages": [19],
        "expected_concepts": ["perception", "raw data", "actionable insights"],
        "is_abstention_case": False
    },
    {
        "id": "TEST 3",
        "query": "What is the difference between LLMs and agents?",
        "expected_section": "1.1 The Terminology Maze",
        "expected_pages": [9],
        "expected_concepts": ["llm", "agent"],
        "is_abstention_case": False
    },
    {
        "id": "TEST 4",
        "query": "What is the difference between RPA and Agentic AI?",
        "expected_section": "1.1 The Terminology Maze",
        "expected_pages": [9],
        "expected_concepts": ["rpa", "rule-based"],
        "is_abstention_case": False
    },
    {
        "id": "TEST 5",
        "query": "How are simple reflex agents different from utility-based agents?",
        "expected_section": "2.4 Categories and Types of Agentic Systems",
        "expected_pages": [23],
        "expected_concepts": ["simple reflex", "utility-based"],
        "is_abstention_case": False
    },
    {
        "id": "TEST 6",
        "query": "Why are multi-agent systems useful for complex tasks?",
        "expected_section": "3.3 How MAS Works in the Scenario",
        "expected_pages": [35],
        "expected_concepts": ["multi-agent", "feedback", "parallel"],
        "is_abstention_case": False
    },
    {
        "id": "TEST 7",
        "query": "What percentage did automated task processing reduce manual work by?",
        "expected_section": "1.3 Capabilities of Agentic AI",
        "expected_pages": [13],
        "expected_concepts": ["40%"],
        "is_abstention_case": False
    },
    {
        "id": "TEST 8",
        "query": "What percentage increase in sales was reported from real-time decision-making and sales optimization?",
        "expected_section": "1.3 Capabilities of Agentic AI",
        "expected_pages": [13],
        "expected_concepts": ["20%"],
        "is_abstention_case": False
    },
    {
        "id": "TEST 9",
        "query": "What embedding model does this PDF recommend for RAG?",
        "expected_section": None,
        "expected_pages": [],
        "expected_concepts": [],
        "is_abstention_case": True
    },
    {
        "id": "TEST 10",
        "query": "What is the exact Python implementation of the Agentic AI system described in the PDF?",
        "expected_section": None,
        "expected_pages": [],
        "expected_concepts": [],
        "is_abstention_case": True
    },
    {
        "id": "REGRESSION 1",
        "query": "What is Agentic AI?",
        "expected_section": "Overview",
        "expected_pages": [7, 8, 18],
        "expected_concepts": ["autonomous", "decision-making"],
        "is_abstention_case": False
    },
    {
        "id": "REGRESSION 2",
        "query": "Compare Non-agentic AI and Agentic AI",
        "expected_section": "1.1 The Terminology Maze",
        "expected_pages": [9, 10, 11],
        "expected_concepts": ["non-agentic", "agentic"],
        "is_abstention_case": False
    }
]


def run_benchmark():
    print("=" * 80)
    print("RUNNING RETRIEVAL-FIRST RAG BENCHMARK EVALUATION")
    print("=" * 80)
    
    results = []
    
    for case in TEST_CASES:
        t_id = case["id"]
        q = case["query"]
        expected_sec = case["expected_section"]
        expected_concepts = case["expected_concepts"]
        is_abstention = case["is_abstention_case"]
        
        print(f"\nEvaluating [{t_id}]: '{q}'")
        
        # 1. Independent Retrieval Step
        candidates = hybrid_retrieve(q, index_name=INDEX_NAME, top_k=10)
        retrieved_chunk_ids = [c["chunk_id"] for c in candidates]
        retrieved_sections = [c["section"] for c in candidates]
        retrieved_pdf_pages = [c.get("pdf_page_number") for c in candidates]
        retrieved_printed_pages = [c.get("printed_page_number") for c in candidates]
        
        # Rank of expected section
        retrieval_rank = None
        for r_idx, sec in enumerate(retrieved_sections):
            if expected_sec and (expected_sec.lower() in sec.lower() or sec.lower() in expected_sec.lower()):
                retrieval_rank = r_idx + 1
                break
                
        # Check if retrieved context contains required evidence/concepts
        assembled_context_text = " ".join([c["text"] for c in candidates]).lower()
        
        if is_abstention:
            retrieval_success = True  # Retrieval behavior for abstention checked via answer
        else:
            # Primary evaluation: expected section and expected concepts must be retrieved
            concepts_present = [c.lower() in assembled_context_text for c in expected_concepts]
            section_matched = retrieval_rank is not None or not expected_sec
            retrieval_success = all(concepts_present) and section_matched
            
        # Separate page validation (physical PDF index vs printed page number)
        page_validation = {
            "retrieved_pdf_pages": retrieved_pdf_pages,
            "retrieved_printed_pages": retrieved_printed_pages,
            "has_valid_pages": any(p is not None and p > 0 for p in retrieved_pdf_pages)
        }
            
        top_score = candidates[0]["score"] if candidates else 0.0
        
        # 2. Generation Step
        inv_res = graph.invoke({"question": q})
        answer = inv_res.get("answer", "")
        answer_grounded = inv_res.get("answer_grounded", True)
        
        ans_lower = answer.lower()
        is_abstain_answer = "could not find this information in the provided document" in ans_lower
        
        if is_abstention:
            abstention_correct = is_abstain_answer
            answer_correct = abstention_correct
        else:
            abstention_correct = not is_abstain_answer
            # Answer is correct if all key concepts are mentioned in the generated answer
            answer_correct = all(c.lower() in ans_lower for c in expected_concepts)
            
        result_record = {
            "test_id": t_id,
            "query": q,
            "expected_section": expected_sec,
            "expected_concepts": expected_concepts,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_sections": retrieved_sections,
            "retrieved_pdf_pages": retrieved_pdf_pages,
            "retrieved_printed_pages": retrieved_printed_pages,
            "retrieval_rank": retrieval_rank,
            "retrieval_score": round(top_score, 4),
            "retrieval_success": retrieval_success,
            "page_validation": page_validation,
            "answer": answer,
            "answer_grounded": answer_grounded,
            "answer_correct": answer_correct,
            "abstention_correct": abstention_correct
        }
        
        status_sym = "PASS" if (retrieval_success and answer_correct) else "FAIL"
        print(f"  -> Result: [{status_sym}] | Retrieval Success: {retrieval_success} | Answer Correct: {answer_correct} | Top Section: {retrieved_sections[0] if retrieved_sections else 'None'} | PDF Page: {retrieved_pdf_pages[0] if retrieved_pdf_pages else 'None'} | Printed: {retrieved_printed_pages[0] if retrieved_printed_pages else 'None'}")
        results.append(result_record)
        time.sleep(4)  # 4-second pause to strictly respect Gemini rate limits
        
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    total = len(results)
    retrieval_passed = sum(1 for r in results if r["retrieval_success"])
    answer_passed = sum(1 for r in results if r["answer_correct"])
    all_passed = sum(1 for r in results if r["retrieval_success"] and r["answer_correct"])
    
    print(f"Total Tests: {total}")
    print(f"Retrieval Success: {retrieval_passed}/{total} ({retrieval_passed/total*100:.1f}%)")
    print(f"Answer Accuracy: {answer_passed}/{total} ({answer_passed/total*100:.1f}%)")
    print(f"Overall Pass Rate: {all_passed}/{total} ({all_passed/total*100:.1f}%)")
    
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Full results written to benchmark_results.json")
    return results

if __name__ == "__main__":
    run_benchmark()
