"""
YojanaGraph — Accuracy Evaluation Script
Runs LLM-as-Judge + BERTScore against all 3 pipelines.

SETUP (run once in WSL):
    pip install requests evaluate bert_score huggingface_hub --break-system-packages

USAGE:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   # get free token at huggingface.co/settings/tokens
    python eval.py

If your dashboard API endpoint is different, change BASE_URL below.
"""

import os
import json
import time
import requests
import evaluate
from huggingface_hub import InferenceClient

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8501"   # change if your API is on a different port
HF_TOKEN = os.environ.get("HF_TOKEN", "")
RESULTS_FILE = "eval_results.json"

# ── Ground truth QA pairs (Indian Govt Schemes domain) ───────────────────────
GROUND_TRUTH = [
    {
        "question": "What is the PM-Vidyalaxmi Scheme?",
        "correct_answer": "PM-Vidyalaxmi is a government scheme that provides financial support for higher education, covering students of all genders pursuing studies at recognized institutions."
    },
    {
        "question": "Who is eligible for the AICTE Pragati Scholarship?",
        "correct_answer": "The AICTE Pragati Scholarship is available for female students pursuing technical education at AICTE-approved institutions."
    },
    {
        "question": "What is the Kanya Utthan Yojana scheme about?",
        "correct_answer": "Kanya Utthan Yojana is a scheme aimed at empowering girl students by providing financial assistance for their education and overall development."
    },
    {
        "question": "What does the Post-Matric Scholarship Scheme provide?",
        "correct_answer": "The Post-Matric Scholarship Scheme provides financial assistance to students from Scheduled Caste, Scheduled Tribe, and Denotified Communities for post-matriculation education, administered by the Ministry of Social Justice and Empowerment."
    },
    {
        "question": "Which schemes are available specifically for Scheduled Caste (SC) students?",
        "correct_answer": "Schemes available for SC students include the Post-Matric Scholarship Scheme and PM-Vidyalaxmi Scheme, among others targeted at marginalized communities."
    },
    {
        "question": "What is the National Scholarship Portal (NSP)?",
        "correct_answer": "The National Scholarship Portal is a digital platform that enables students to apply for various government scholarships online, consolidating multiple scholarship schemes in one place."
    },
    {
        "question": "What is the Ayushman Bharat scheme?",
        "correct_answer": "Ayushman Bharat is a health insurance scheme providing free healthcare coverage to economically vulnerable families, including Scheduled Caste and Scheduled Tribe communities."
    },
    {
        "question": "What is the purpose of the Scholarship for Top Class Education for Students with Disabilities (STC)?",
        "correct_answer": "The STC scheme provides financial assistance to students with disabilities, including girls, to pursue higher education at top institutions."
    },
    {
        "question": "What is Kishore Vaigyanik Protsahan Yojana (KVPY)?",
        "correct_answer": "KVPY is a national scholarship scheme that supports students pursuing undergraduate and postgraduate studies in science, identifying and nurturing scientific talent."
    },
    {
        "question": "Which government scheme provides free healthcare to the poor and marginalized communities?",
        "correct_answer": "Ayushman Bharat – Health and Wellness Centres provides free healthcare facilities to the poor, particularly to Scheduled Tribe and Scheduled Caste communities."
    },
    {
        "question": "What kind of support does the Mahila Shakti Scheme provide?",
        "correct_answer": "The Mahila Shakti Scheme is designed to empower women in various fields including business, entrepreneurship, and education."
    },
    {
        "question": "What does the Nalanda Scheme focus on?",
        "correct_answer": "The Nalanda Scheme is a National Initiative for School Heads and Teachers (NISHTHA) that improves the quality of government schools and provides financial assistance to girls from the Scheduled Caste community."
    },
    {
        "question": "Which scheme supports women from the SC community in pursuing education and entrepreneurship?",
        "correct_answer": "The Savitribai Phule Mahila Shakti Kendras scheme provides training and support to women from the SC community to pursue education, entrepreneurship, and other skills."
    },
    {
        "question": "What is the Lok Jumbish Scheme?",
        "correct_answer": "The Lok Jumbish Scheme aims to improve the quality of education for Scheduled Caste students, especially girls, providing financial assistance for education, infrastructure development, and teacher training."
    },
    {
        "question": "Which schemes are available for female students in India?",
        "correct_answer": "Schemes available for female students include AICTE Pragati Scholarship, Kanya Utthan Yojana, PM-Vidyalaxmi Scheme, and various state-level scholarships targeting girl students."
    },
]

# ── Call your dashboard API ──────────────────────────────────────────────────
def query_pipelines(question: str) -> dict:
    """
    Calls the dashboard backend and returns answers from all 3 pipelines.
    Adjust the endpoint path and payload to match your actual API.
    Common patterns tried automatically.
    """
    endpoints_to_try = [
        f"{BASE_URL}/query",
        f"{BASE_URL}/api/query",
        f"{BASE_URL}/compare",
        f"{BASE_URL}/api/compare",
    ]
    payload = {"query": question, "question": question}

    for endpoint in endpoints_to_try:
        try:
            resp = requests.post(endpoint, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                # Try common response shapes
                return {
                    "llm_only": (
                        data.get("llm_only") or
                        data.get("llm") or
                        data.get("pipeline1") or
                        data.get("results", {}).get("llm_only", "")
                    ),
                    "basic_rag": (
                        data.get("basic_rag") or
                        data.get("rag") or
                        data.get("pipeline2") or
                        data.get("results", {}).get("basic_rag", "")
                    ),
                    "graphrag": (
                        data.get("graphrag") or
                        data.get("graph_rag") or
                        data.get("pipeline3") or
                        data.get("results", {}).get("graphrag", "")
                    ),
                }
        except Exception:
            continue

    # If API not reachable, return empty — you'll fill manually
    print(f"  [WARNING] Could not reach API for: {question[:50]}")
    return {"llm_only": "", "basic_rag": "", "graphrag": ""}


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────
JUDGE_PROMPT = """Grade the system's answer strictly.
Question: {q}
Correct answer: {correct}
System answer: {answer}

Reply with only PASS or FAIL.
PASS = the system answer correctly addresses the question with no major errors or hallucinations.
FAIL = the answer is wrong, missing key facts, or contradicts the correct answer."""

def llm_judge(client, question, correct_answer, system_answer):
    if not system_answer or not system_answer.strip():
        return False
    try:
        prompt = JUDGE_PROMPT.format(
            q=question,
            correct=correct_answer,
            answer=system_answer
        )
        verdict = client.chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        result = verdict.choices[0].message.content.upper()
        return "PASS" in result
    except Exception as e:
        print(f"  [Judge error] {e}")
        return False


# ── Main evaluation ───────────────────────────────────────────────────────────
def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set.")
        print("Get a free token at https://huggingface.co/settings/tokens")
        print("Then run: export HF_TOKEN=hf_xxxx")
        return

    print("Setting up LLM judge (Llama-3.1-8B via HuggingFace)...")
    client = InferenceClient(
        model="meta-llama/Llama-3.1-8B-Instruct",
        token=HF_TOKEN
    )

    print("Loading BERTScore...")
    bertscore = evaluate.load("bertscore")

    # Collect pipeline outputs
    llm_answers, rag_answers, graph_answers = [], [], []

    print(f"\nQuerying all 3 pipelines for {len(GROUND_TRUTH)} questions...\n")
    for i, item in enumerate(GROUND_TRUTH):
        q = item["question"]
        print(f"[{i+1}/{len(GROUND_TRUTH)}] {q[:60]}...")
        result = query_pipelines(q)
        llm_answers.append(result["llm_only"] or "No answer")
        rag_answers.append(result["basic_rag"] or "No answer")
        graph_answers.append(result["graphrag"] or "No answer")
        time.sleep(0.5)  # be gentle on local server

    references = [item["correct_answer"] for item in GROUND_TRUTH]
    questions   = [item["question"] for item in GROUND_TRUTH]

    print("\nRunning LLM-as-Judge evaluation...")
    llm_judge_results, rag_judge_results, graph_judge_results = [], [], []

    for i, (q, ref) in enumerate(zip(questions, references)):
        print(f"  Judging Q{i+1}...")
        llm_judge_results.append(llm_judge(client, q, ref, llm_answers[i]))
        rag_judge_results.append(llm_judge(client, q, ref, rag_answers[i]))
        graph_judge_results.append(llm_judge(client, q, ref, graph_answers[i]))
        time.sleep(0.3)

    print("\nRunning BERTScore...")
    def bert_f1(predictions):
        scores = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            rescale_with_baseline=True
        )
        return sum(scores["f1"]) / len(scores["f1"])

    llm_bert   = bert_f1(llm_answers)
    rag_bert   = bert_f1(rag_answers)
    graph_bert = bert_f1(graph_answers)

    # ── Results ───────────────────────────────────────────────────────────────
    results = {
        "questions_evaluated": len(GROUND_TRUTH),
        "llm_only": {
            "llm_judge_pass_rate": round(sum(llm_judge_results) / len(llm_judge_results), 3),
            "llm_judge_passes": sum(llm_judge_results),
            "bertscore_f1_rescaled": round(llm_bert, 3),
        },
        "basic_rag": {
            "llm_judge_pass_rate": round(sum(rag_judge_results) / len(rag_judge_results), 3),
            "llm_judge_passes": sum(rag_judge_results),
            "bertscore_f1_rescaled": round(rag_bert, 3),
        },
        "graphrag": {
            "llm_judge_pass_rate": round(sum(graph_judge_results) / len(graph_judge_results), 3),
            "llm_judge_passes": sum(graph_judge_results),
            "bertscore_f1_rescaled": round(graph_bert, 3),
        },
        "raw_answers": {
            "llm_only": llm_answers,
            "basic_rag": rag_answers,
            "graphrag": graph_answers,
        }
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"{'Metric':<30} {'LLM-Only':>10} {'Basic RAG':>10} {'GraphRAG':>10}")
    print("-"*60)
    print(f"{'LLM-Judge pass rate':<30} {results['llm_only']['llm_judge_pass_rate']:>10.1%} {results['basic_rag']['llm_judge_pass_rate']:>10.1%} {results['graphrag']['llm_judge_pass_rate']:>10.1%}")
    print(f"{'LLM-Judge passes (/{len(GROUND_TRUTH)})':<30} {results['llm_only']['llm_judge_passes']:>10} {results['basic_rag']['llm_judge_passes']:>10} {results['graphrag']['llm_judge_passes']:>10}")
    print(f"{'BERTScore F1 (rescaled)':<30} {results['llm_only']['bertscore_f1_rescaled']:>10.3f} {results['basic_rag']['bertscore_f1_rescaled']:>10.3f} {results['graphrag']['bertscore_f1_rescaled']:>10.3f}")
    print("="*60)
    print(f"\nFull results saved to: {RESULTS_FILE}")
    print("Paste the numbers above into your benchmark report.")

    # Bonus threshold check
    g = results["graphrag"]
    print("\n── Bonus threshold check (GraphRAG) ──")
    print(f"  LLM-Judge ≥ 90%:        {'✅ YES' if g['llm_judge_pass_rate'] >= 0.9 else '❌ NO'} ({g['llm_judge_pass_rate']:.1%})")
    print(f"  BERTScore F1 ≥ 0.55:    {'✅ YES' if g['bertscore_f1_rescaled'] >= 0.55 else '❌ NO'} ({g['bertscore_f1_rescaled']:.3f})")


if __name__ == "__main__":
    main()