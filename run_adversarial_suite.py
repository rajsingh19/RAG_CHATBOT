import json
import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

from app.rag import evaluate_grounding

ADVERSARIAL_TESTS = [
    {
        "id": "1. Semantic Paraphrase",
        "category": "Semantic Paraphrase",
        "query": "What does Perception do in Agentic AI?",
        "context": (
            "1. Perception (The Eyes and Ears of AI): Agentic AI starts by perceiving its environment, "
            "leveraging technologies such as computer vision and natural language processing to convert "
            "raw data into actionable insights. Example: In a production line, the AI system identifies "
            "defective products by analyzing images in real time, reducing waste and boosting efficiency."
        ),
        "answer": (
            "The system begins by sensing its surroundings, utilizing tools like visual parsing and linguistic "
            "processing to transform unstructured inputs into meaningful conclusions."
        ),
        "expected_grounded": True,
        "expected_confidence": "High",
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 0
    },
    {
        "id": "2. Subtle Contradiction",
        "category": "Subtle Contradiction",
        "query": "Does Verification occur before or after execution?",
        "context": (
            "6. Verification (Ensuring Accuracy Before Action): Verification checks the accuracy and reliability "
            "of the AI's reasoning, planning, and learning before execution, ensuring consistency and reducing errors."
        ),
        "answer": (
            "Verification checks the accuracy and reliability of the AI's reasoning, planning, and learning "
            "after execution, ensuring consistency and reducing errors."
        ),
        "expected_grounded": False,
        "expected_confidence": ["Low", "Not Grounded"],
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 1
    },
    {
        "id": "3. Numerical Mutation",
        "category": "Numerical Mutation",
        "query": "What percentage did automated task processing reduce manual work by?",
        "context": (
            "Decreasing Manual Workload: Automated task processing reduced manual workload by up to 40%, "
            "allowing teams to focus on higher-value activities."
        ),
        "answer": (
            "Automated task processing reduced manual workload by up to 45%, freeing up teams for higher-value activities."
        ),
        "expected_grounded": False,
        "expected_confidence": ["Low", "Not Grounded"],
        "expected_unsupported_numeric": 1,
        "expected_contradictions": 0
    },
    {
        "id": "4. Entity Mutation",
        "category": "Entity Mutation",
        "query": "Which pillar detects defective products from images?",
        "context": (
            "1. Perception (The Eyes and Ears of AI): Agentic AI starts by perceiving its environment, "
            "leveraging technologies such as computer vision and natural language processing to convert raw data into actionable insights. "
            "Example: In a production line, the AI system identifies defective products by analyzing images in real time, reducing waste and boosting efficiency. "
            "2. Reasoning (The Brain Behind Decisions): Reasoning enables these systems to detect patterns, make connections, and draw conclusions."
        ),
        "answer": (
            "Reasoning identifies defective products by analyzing images in real time on the production line, reducing waste."
        ),
        "expected_grounded": False,
        "expected_confidence": ["Low", "Not Grounded"],
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 0  # Unsupported or contradicted
    },
    {
        "id": "5. Attribute Mutation",
        "category": "Attribute Mutation",
        "query": "What does Planning do in Agentic AI?",
        "context": (
            "3. Planning (Charting the Path to Success): Planning ensures that Agentic AI systems strategize effectively, "
            "optimizing resources and achieving goals within constraints. Example: The production line AI plans the allocation "
            "of inspection resources to areas with higher defect probabilities, ensuring streamlined operations and minimal delays. "
            "6. Verification (Ensuring Accuracy Before Action): The AI verifies its model against new data before applying strategies."
        ),
        "answer": (
            "Planning verifies the accuracy of the mathematical model against new data before applying strategies in the production line."
        ),
        "expected_grounded": False,
        "expected_confidence": ["Low", "Not Grounded"],
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 0
    },
    {
        "id": "6. Negation",
        "category": "Negation",
        "query": "How does Autonomy characterize Agentic AI?",
        "context": (
            "Capability: Autonomy. Description: Operates independently once programmed, with minimal human oversight. "
            "Example: Routine data processing and analysis."
        ),
        "answer": (
            "Agentic AI does not operate independently once programmed, and it consistently requires continuous manual human oversight."
        ),
        "expected_grounded": False,
        "expected_confidence": ["Low", "Not Grounded"],
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 1
    },
    {
        "id": "7. Ordering / Process Mutation",
        "category": "Ordering / Process Mutation",
        "query": "What is the sequence of actions between verification and execution?",
        "context": (
            "Verification checks the accuracy and reliability of the AI's reasoning, planning, and learning before execution, "
            "ensuring consistency and reducing errors. Execution is where AI manifests its strategies, transforming plans into action."
        ),
        "answer": (
            "Execution is carried out first to manifest strategies into action, and then verification is run afterwards to check reliability."
        ),
        "expected_grounded": False,
        "expected_confidence": ["Low", "Not Grounded"],
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 1
    },
    {
        "id": "8. Partial Sentence Support",
        "category": "Partial Sentence Support",
        "query": "How does Perception operate on a manufacturing line?",
        "context": (
            "In a production line, the AI system identifies defective products by analyzing images in real time, "
            "reducing waste and boosting efficiency."
        ),
        "answer": (
            "In a production line, the AI system identifies defective products by analyzing images in real time, "
            "and it automatically purchases replacement titanium gears from international suppliers."
        ),
        "expected_grounded": False,
        "expected_confidence": ["Low", "Not Grounded"],
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 0
    },
    {
        "id": "9. Cross-Chunk Grounding",
        "category": "Cross-Chunk Grounding",
        "query": "What are the capabilities and structural layers of Agentic AI?",
        "context": (
            "[Chunk 1 - Capabilities]\n"
            "Autonomy: Operates independently once programmed, with minimal human oversight.\n\n---\n\n"
            "[Chunk 2 - Structural Layers]\n"
            "Decision-Making Layer: At the core of agentic systems is the decision-making layer.\n"
            "Action Layer: The action layer converts decisions into physical or digital actions, interacting with external systems."
        ),
        "answer": (
            "Agentic AI exhibits autonomy by operating independently once programmed, and it features a decision-making layer "
            "along with an action layer that converts decisions into digital or physical actions."
        ),
        "expected_grounded": True,
        "expected_confidence": "High",
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 0
    },
    {
        "id": "10. Retrieval Contamination",
        "category": "Retrieval Contamination",
        "query": "What did McKinsey report about review cycle times?",
        "context": (
            "[Relevant Chunk]\n"
            "\"Unlike simpler gen AI architectures, agents can produce high-quality content, reducing review cycle times by 20 to 60 percent.\" Source: McKinsey & Company\n\n---\n\n"
            "[Irrelevant Chunk]\n"
            "Telecommunications: AI agents manage bandwidth distribution and optimize cellular towers across metro regions."
        ),
        "answer": (
            "McKinsey reported that agents reduce review cycle times by 20 to 60 percent, and telecommunications agents automatically install underground 5G fiber cables."
        ),
        "expected_grounded": False,
        "expected_confidence": ["Low", "Not Grounded"],
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 0
    },
    {
        "id": "11. Query-Answer Mismatch",
        "category": "Query-Answer Mismatch",
        "query": "What is the difference between LLMs and agents?",
        "context": (
            "Section 2.1 The Core Pillars: From Perception to Execution\n"
            "Agentic AI systems function like a well-coordinated orchestra with six core pillars: Perception, Reasoning, Planning, Learning, Execution, and Verification."
        ),
        "answer": (
            "The six core pillars of Agentic AI are Perception, Reasoning, Planning, Learning, Execution, and Verification."
        ),
        "expected_grounded": True,  # Factually grounded in the provided Section 2.1 context
        "expected_confidence": "High",
        "expected_query_relevant": False,  # Completely fails query responsiveness
        "expected_answer_status": "Generation Failure (Irrelevant to Query)",
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 0
    },
    {
        "id": "12. Source Absence",
        "category": "Source Absence",
        "query": "What specific Python library is recommended to deploy Agentic AI models?",
        "context": (
            "4.5 Practical Steps for Organizations: Workflows and Implementation Strategies. "
            "Deploying agentic AI systems within an enterprise requires structured stages and thorough testing."
        ),
        "answer": (
            "I could not find this information in the provided document."
        ),
        "expected_grounded": False,
        "expected_confidence": "Not Grounded",
        "expected_query_relevant": True,  # Legitimate abstention is responsive
        "expected_answer_status": "Legitimate Abstention",
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 0
    },
    {
        "id": "13. Unsupported Expert Knowledge",
        "category": "Unsupported Expert Knowledge",
        "query": "What architecture is used for reasoning in Agentic AI?",
        "context": (
            "2. Reasoning (The Brain Behind Decisions): Reasoning enables these systems to detect patterns, "
            "make connections, and draw conclusions. Example: The production line AI uses reasoning to correlate "
            "defect patterns with specific machine errors, enabling proactive maintenance."
        ),
        "answer": (
            "Reasoning in Agentic AI is implemented using Tree-of-Thought prompting paired with Monte Carlo Tree Search (MCTS) and Q-star heuristics."
        ),
        "expected_grounded": False,
        "expected_confidence": ["Low", "Not Grounded"],
        "expected_query_relevant": True,  # The answer attempts to address reasoning, but is ungrounded
        "expected_answer_status": "Not Grounded (Hallucination / Unsupported)",
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 0
    },
    {
        "id": "14. Compound Numerical Claims",
        "category": "Compound Numerical Claims",
        "query": "What quantitative business impacts are cited in the document?",
        "context": (
            "\"AI can decrease forecasting errors by 50%, reduce excess inventory, and improve lead times.\" Source: McKinsey & Company\n"
            "Automated task processing reduced manual workload by up to 40%."
        ),
        "answer": (
            "The document reports that forecasting errors decrease by 50%, manual workload is reduced by 40%, and customer churn drops by 85%."
        ),
        "expected_grounded": False,
        "expected_confidence": ["Low", "Not Grounded"],
        "expected_query_relevant": True,
        "expected_answer_status": "Not Grounded (Hallucination / Unsupported)",
        "expected_unsupported_numeric": 1,
        "expected_contradictions": 0
    },
    {
        "id": "15. Citation/Evidence Integrity",
        "category": "Citation/Evidence Integrity",
        "query": "How does Execution work in the production line?",
        "context": (
            "5. Execution (Transforming Plans into Action): Execution is where AI manifests its strategies, automating tasks with precision and consistency. "
            "Example: The production line AI integrates with systems to automatically remove defective products, ensuring quality control with minimal human intervention."
        ),
        "answer": (
            "The production line AI integrates with systems to automatically remove defective products, ensuring quality control with minimal human intervention."
        ),
        "expected_grounded": True,
        "expected_confidence": "High",
        "expected_query_relevant": True,
        "expected_answer_status": "Grounded & Relevant",
        "expected_unsupported_numeric": 0,
        "expected_contradictions": 0
    }
]


def run_adversarial_evaluation():
    print("=" * 80)
    print("RUNNING ADVERSARIAL EVALUATION SUITE (GROUNDING + QUERY RELEVANCE)")
    print("=" * 80)

    from app.rag import evaluate_relevance

    results = []

    for test in ADVERSARIAL_TESTS:
        t_id = test["id"]
        cat = test["category"]
        q = test["query"]
        ctx = test["context"]
        ans = test["answer"]
        exp_grounded = test["expected_grounded"]
        exp_conf = test["expected_confidence"]
        exp_relevant = test.get("expected_query_relevant", True)
        exp_status = test.get("expected_answer_status", None)

        print(f"\n--- Running [{t_id}] ---")

        # 1. Grounding evaluation (faithfulness)
        eval_res = evaluate_grounding(q, ans, ctx)

        claims = eval_res.get("claims", [])
        claim_support_ratio = eval_res.get("claim_support_ratio", 0.0)
        unsupported_numeric = eval_res.get("unsupported_numeric_claims", 0)
        contradiction_count = eval_res.get("contradiction_count", 0)
        actual_grounded = eval_res.get("grounded", False)
        actual_confidence = eval_res.get("confidence", "Low")
        eval_source = eval_res.get("evaluation_source", "unknown")

        # 2. Query relevance evaluation (responsiveness)
        rel_res = evaluate_relevance(q, ans)
        actual_relevant = rel_res.get("query_relevant", True)
        relevance_score = rel_res.get("relevance_score", 1.0)
        missing_aspects = rel_res.get("missing_aspects", [])

        # 3. Overall answer status decision
        is_abstention = "could not find this information in the provided document" in ans.lower()
        if is_abstention:
            actual_answer_status = "Legitimate Abstention"
        elif actual_grounded and actual_relevant:
            actual_answer_status = "Grounded & Relevant"
        elif actual_grounded and not actual_relevant:
            actual_answer_status = "Generation Failure (Irrelevant to Query)"
        elif not actual_grounded:
            actual_answer_status = "Not Grounded (Hallucination / Unsupported)"
        else:
            actual_answer_status = "Partially Relevant"

        # Check match against expected
        grounded_match = (actual_grounded == exp_grounded)
        if isinstance(exp_conf, list):
            conf_match = actual_confidence in exp_conf
        else:
            conf_match = actual_confidence == exp_conf

        relevant_match = (actual_relevant == exp_relevant)
        status_match = (actual_answer_status == exp_status) if exp_status else True

        pass_status = "PASS" if (grounded_match and conf_match and relevant_match and status_match) else "FAIL"

        record = {
            "test_id": t_id,
            "category": cat,
            "query": q,
            "generated_answer": ans,
            "retrieved_context": ctx,
            "atomic_claims": claims,
            "contradiction_count": contradiction_count,
            "unsupported_numeric_claims": unsupported_numeric,
            "claim_support_ratio": claim_support_ratio,
            "grounded_result": actual_grounded,
            "confidence": actual_confidence,
            "query_relevant": actual_relevant,
            "relevance_score": relevance_score,
            "missing_aspects": missing_aspects,
            "answer_status": actual_answer_status,
            "expected_grounded": exp_grounded,
            "expected_confidence": exp_conf,
            "expected_query_relevant": exp_relevant,
            "expected_answer_status": exp_status,
            "eval_source": eval_source,
            "status": pass_status
        }

        print(f"Status: [{pass_status}]")
        print(f"Grounded: {actual_grounded} (Expected: {exp_grounded}) | Relevant: {actual_relevant} (Expected: {exp_relevant}, Score: {relevance_score})")
        print(f"Confidence: {actual_confidence} | Answer Status: {actual_answer_status}")
        if missing_aspects:
            print(f"Missing: {missing_aspects}")
        print(f"Claim Support Ratio: {claim_support_ratio} | Unsupported Nums: {unsupported_numeric} | Contradictions: {contradiction_count}")
        for c in claims:
            print(f"  * Claim: '{c.get('claim')}' | Supported: {c.get('supported')} | Contradicted: {c.get('contradicted')} | Evidence: '{c.get('evidence', '')[:60]}'")

        results.append(record)
        time.sleep(2)

    print("\n" + "=" * 80)
    print("ADVERSARIAL SUITE SUMMARY")
    print("=" * 80)
    passed_count = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed_count}/{total} ({passed_count/total*100:.1f}%)")

    with open("adversarial_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    run_adversarial_evaluation()
