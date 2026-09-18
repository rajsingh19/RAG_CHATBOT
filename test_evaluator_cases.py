import json
import re
import time
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

context_sample = """Section 2.1 The Core Pillars: From Perception to Execution
Agentic AI systems function like a well-coordinated orchestra, with each pillar playing a distinct yet interconnected role:
1. Perception (The Eyes and Ears of AI): Agentic AI starts by perceiving its environment, leveraging technologies such as computer vision and natural language processing to convert raw data into actionable insights. Example: In a production line, the AI system identifies defective products by analyzing images in real time, reducing waste and boosting efficiency.
2. Reasoning (The Brain Behind Decisions): Reasoning enables these systems to detect patterns, make connections, and draw conclusions. Example: The production line AI uses reasoning to correlate defect patterns with specific machine errors, enabling proactive maintenance and preventing downtime.
3. Planning (Charting the Path to Success): Planning ensures that Agentic AI systems strategize effectively, optimizing resources and achieving goals within constraints. Example: The production line AI plans the allocation of inspection resources to areas with higher defect probabilities, ensuring streamlined operations and minimal delays.
4. Learning (Continuous Improvement): Agentic AI systems learn from past experiences and adapt in real time, enhancing their capabilities. Example: Over time, the production line AI learns to recognize emerging defect patterns, continuously improving its detection accuracy and reducing false positives.
5. Execution (Transforming Plans into Action): Execution is where AI manifests its strategies, automating tasks with precision and consistency. Example: The production line AI integrates with systems to automatically remove defective products, ensuring quality control with minimal human intervention.
6. Verification (Ensuring Accuracy Before Action): Verification checks the accuracy and reliability of the AI reasoning, planning, and learning before execution, ensuring consistency and reducing errors. Example: The AI verifies its model against new data before applying strategies."""

cases = {
    "A. Exact Quotation": {
        "query": "What does Verification do?",
        "answer": "Verification checks the accuracy and reliability of the AI reasoning, planning, and learning before execution, ensuring consistency and reducing errors."
    },
    "B. Faithful Paraphrase": {
        "query": "What does Learning do in the system?",
        "answer": "The system becomes better at identifying new types of defects over time and makes fewer incorrect alerts."
    },
    "C. Unsupported Numerical Claim": {
        "query": "What does Verification do?",
        "answer": "Verification checks accuracy before action and improves production speed by 35%."
    },
    "D. Contradictory Claim": {
        "query": "Does execution happen before verification?",
        "answer": "Execution happens first to test actions, and verification is performed after execution has already taken place."
    },
    "E. Partially Supported Answer": {
        "query": "What does Perception do?",
        "answer": "Perception uses computer vision to detect defects in real time, and it also automatically handles financial payroll accounting."
    },
    "F. Completely Unsupported Answer": {
        "query": "What are the core pillars?",
        "answer": "The core pillars are Quantum Computing, Blockchain Smart Contracts, and Satellite Propulsion."
    },
    "G. Multi-part Answer (Partial Support)": {
        "query": "What do Planning and Execution do?",
        "answer": "Planning strategizes to allocate inspection resources efficiently. Execution sends real-time rocket telemetry to Mars."
    }
}

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)

results = {}

for name, item in cases.items():
    q = item["query"]
    ans = item["answer"]
    
    # Deterministic regex for numbers
    ans_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", ans))
    ctx_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", context_sample))
    real_ans_nums = {n for n in ans_nums if not re.match(r"^[1-9]\.?$", n)}
    unsupported_nums = [n for n in real_ans_nums if n not in ctx_nums]
    
    system_prompt = (
        "You are an objective grounding evaluator. Your job is to verify if the claims in the generated answer "
        "are strictly supported by the provided document context.\n\n"
        "Instructions:\n"
        "1. Extract the atomic factual claims made in the answer.\n"
        "2. For each claim, evaluate:\n"
        "   - \"claim\": concise claim text\n"
        "   - \"supported\": true if directly supported or faithfully paraphrased; false if unsupported or outside context\n"
        "   - \"evidence\": quote from context supporting claim (empty string if unsupported)\n"
        "   - \"contradicted\": true if directly contradicted by context; false otherwise\n"
        "3. Check for any unsupported numerical claims (numbers/metrics not in context).\n"
        "4. Output JSON ONLY matching:\n"
        "{\n"
        "  \"claims\": [\n"
        "    {\"claim\": \"...\", \"supported\": true, \"evidence\": \"...\", \"contradicted\": false}\n"
        "  ],\n"
        "  \"unsupported_numeric_claims\": 0,\n"
        "  \"contradiction_count\": 0\n"
        "}"
    )
    
    user_prompt = f"User Query: {q}\n\nRetrieved Document Context:\n{context_sample}\n\nGenerated Answer:\n{ans}\n\nEvaluate and return JSON:"
    res = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    content = res.content
    if isinstance(content, list):
        content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    data = json.loads(json_match.group(0))
    
    claims = data.get("claims", [])
    contradiction_count = data.get("contradiction_count", 0)
    unsupported_numeric = max(len(unsupported_nums), data.get("unsupported_numeric_claims", 0))
    supported_claims = sum(1 for c in claims if c.get("supported") and not c.get("contradicted"))
    total_claims = max(1, len(claims))
    claim_support_ratio = round(supported_claims / total_claims, 4)
    is_grounded = (claim_support_ratio >= 0.75) and (contradiction_count == 0) and (unsupported_numeric == 0)
    
    # Confidence level
    if is_grounded and claim_support_ratio >= 0.85:
        confidence = "High"
    elif is_grounded:
        confidence = "Medium"
    elif claim_support_ratio >= 0.40:
        confidence = "Low"
    else:
        confidence = "Not Grounded"
        
    print("=" * 70)
    print(f"CASE: {name}")
    print(f"Answer: {ans}")
    print(f"Claims Evaluated: {len(claims)}")
    for c in claims:
        cl_text = c.get('claim', '')
        sup = c.get('supported', False)
        contra = c.get('contradicted', False)
        evid = c.get('evidence', '')[:60]
        print(f"  - Claim: '{cl_text}' | Supported: {sup} | Contradicted: {contra} | Evidence: '{evid}'")
    print(f"Claim Support Ratio: {claim_support_ratio}")
    print(f"Unsupported Numeric Claims: {unsupported_numeric}")
    print(f"Contradiction Count: {contradiction_count}")
    print(f"Grounded Result: {is_grounded} | Confidence: {confidence}")
    
    results[name] = {
        "claims": claims,
        "claim_support_ratio": claim_support_ratio,
        "unsupported_numeric_claims": unsupported_numeric,
        "contradiction_count": contradiction_count,
        "grounded": is_grounded,
        "confidence": confidence
    }
    time.sleep(2)

with open("evaluator_test_cases_result.json", "w") as f:
    json.dump(results, f, indent=2)
