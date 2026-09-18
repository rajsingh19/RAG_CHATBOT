import json
import time
from dotenv import load_dotenv
load_dotenv()

from app.rag import evaluate_grounding, evaluate_relevance

RELEVANCE_TESTS = [
    {
        "id": "TEST R1",
        "query": "What is the difference between LLMs and agents?",
        "answer": "The six core pillars are Perception, Reasoning, Planning, Learning, Execution, and Verification.",
        "context": "Section 2.1 The Core Pillars: From Perception to Execution\nAgentic AI systems function like a well-coordinated orchestra, with each pillar playing a distinct yet interconnected role: Perception, Reasoning, Planning, Learning, Execution, and Verification.",
        "expected_grounded": True,
        "expected_query_relevant": False
    },
    {
        "id": "TEST R2",
        "query": "What is the difference between LLMs and agents?",
        "answer": "LLMs respond to prompts while agents operate toward goals autonomously.",
        "context": "1.1 The Terminology Maze: Unlike LLMs which generate responses when prompted, agents perceive and act toward goals autonomously.",
        "expected_grounded": True,
        "expected_query_relevant": True
    },
    {
        "id": "TEST R3",
        "query": "What does Perception do?",
        "answer": "Planning allocates inspection resources.",
        "context": "1. Perception (The Eyes and Ears of AI): Perception converts raw data into actionable insights.\n3. Planning: Planning allocates inspection resources.",
        "expected_grounded": True,
        "expected_query_relevant": False
    },
    {
        "id": "TEST R4",
        "query": "What are the six core pillars and what does each pillar do?",
        "answer": "Perception, Reasoning, Planning, Learning, Verification and Execution.",
        "context": "Section 2.1 The Core Pillars: From Perception to Execution\n1. Perception converts raw data...\n2. Reasoning detects patterns...\n3. Planning optimizes resources...\n4. Learning continuously improves...\n5. Execution manifests strategies...\n6. Verification checks accuracy...",
        "expected_grounded": True,
        "expected_query_relevant": False  # partial / incomplete
    },
    {
        "id": "TEST R5",
        "query": "What percentage did automated task processing reduce manual work by?",
        "answer": "Agentic AI improves workflow optimization and autonomy.",
        "context": "Decreasing Manual Workload: Automated task processing reduced manual workload by up to 40%.",
        "expected_grounded": False,
        "expected_query_relevant": False
    },
    {
        "id": "TEST R6",
        "query": "What percentage did automated task processing reduce manual work by?",
        "answer": "It reduced manual workload by up to 40%.",
        "context": "Decreasing Manual Workload: Automated task processing reduced manual workload by up to 40%.",
        "expected_grounded": True,
        "expected_query_relevant": True
    },
    {
        "id": "TEST R7",
        "query": "Compare LLMs and agents.",
        "answer": "LLMs primarily handle language understanding and generation.",
        "context": "The table below compares the key aspects of LLMs and agents: LLMs handle language generation while agents act autonomously.",
        "expected_grounded": True,
        "expected_query_relevant": False  # partial, not full comparison
    },
    {
        "id": "TEST R8",
        "query": "What does Verification do?",
        "answer": "Verification checks the accuracy and reliability of reasoning, planning and learning before execution.",
        "context": "6. Verification (Ensuring Accuracy Before Action): Verification checks the accuracy and reliability of the AI's reasoning, planning, and learning before execution, ensuring consistency and reducing errors.",
        "expected_grounded": True,
        "expected_query_relevant": True
    }
]

def run_relevance_tests():
    print("=" * 80)
    print("RUNNING QUERY-ANSWER RELEVANCE SUITE (TESTS R1 - R8)")
    print("=" * 80)

    results = []

    for t in RELEVANCE_TESTS:
        tid = t["id"]
        q = t["query"]
        a = t["answer"]
        ctx = t["context"]
        exp_grounded = t["expected_grounded"]
        exp_relevant = t["expected_query_relevant"]

        # 1. Grounding evaluation (independent)
        g_res = evaluate_grounding(q, a, ctx)
        actual_grounded = g_res["grounded"]

        # 2. Relevance evaluation (independent)
        r_res = evaluate_relevance(q, a)
        actual_relevant = r_res["query_relevant"]
        rel_score = r_res["relevance_score"]
        missing = r_res.get("missing_aspects", [])
        reqs = r_res.get("requirements", [])

        grounded_ok = (actual_grounded == exp_grounded)
        relevant_ok = (actual_relevant == exp_relevant)
        passed = grounded_ok and relevant_ok
        status = "PASS" if passed else "FAIL"

        print(f"\n--- [{tid}]: Status: {status} ---")
        print(f"Query: \"{q}\"")
        print(f"Answer: \"{a}\"")
        print(f"Grounded: {actual_grounded} (Expected: {exp_grounded}) -> {'OK' if grounded_ok else 'DIFF'}")
        print(f"Relevant: {actual_relevant} (Expected: {exp_relevant}) | Score: {rel_score} -> {'OK' if relevant_ok else 'DIFF'}")
        if missing:
            print(f"Missing Aspects: {missing}")

        results.append({
            "id": tid,
            "query": q,
            "answer": a,
            "grounded": actual_grounded,
            "expected_grounded": exp_grounded,
            "query_relevant": actual_relevant,
            "expected_query_relevant": exp_relevant,
            "relevance_score": rel_score,
            "missing_aspects": missing,
            "requirements": reqs,
            "status": status
        })
        time.sleep(2)

    passed_count = sum(1 for r in results if r["status"] == "PASS")
    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed_count}/{len(results)} Passed ({passed_count/len(results)*100:.1f}%)")
    print("=" * 80)

    with open("relevance_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    run_relevance_tests()
