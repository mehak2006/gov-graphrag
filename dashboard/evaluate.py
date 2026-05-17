import json
import evaluate
from bert_score import score

# --------------------------------
# LOAD DATA
# --------------------------------

with open("benchmark_questions.json") as f:
    data = json.load(f)

# --------------------------------
# FAKE SAMPLE OUTPUTS
# Replace with your actual outputs
# --------------------------------

llm_outputs = [
    "Some random schemes.",
    "Minority scholarship.",
    "Engineering scholarship.",
    "SC scholarship.",
    "Girls education schemes."
]

rag_outputs = [
    "AICTE Pragati Scholarship and Kanya Utthan Yojana.",
    "National Scholarship for Minority Students and PM Yasasvi.",
    "AICTE Pragati Scholarship.",
    "Post Matric Scholarship for SC students.",
    "AICTE Pragati Scholarship and Kanya Utthan Yojana."
]

graphrag_outputs = [
    "PM-Vidyalaxmi Scheme.",
    "Minority scholarship.",
    "Engineering scholarship.",
    "SC scholarship.",
    "Girls education support."
]

ground_truth = [x["correct_answer"] for x in data]

# --------------------------------
# SIMPLE PASS RATE
# --------------------------------

def pass_rate(outputs, truths):

    passed = 0

    for o, t in zip(outputs, truths):

        if any(word.lower() in o.lower() for word in t.split()):
            passed += 1

    return passed / len(outputs)

# --------------------------------
# BERT SCORE
# --------------------------------

def bert_f1(outputs, truths):

    P, R, F1 = score(
        outputs,
        truths,
        lang="en"
    )

    return F1.mean().item()

# --------------------------------
# RESULTS
# --------------------------------

print("\nLLM ONLY")
print("Pass Rate:", pass_rate(llm_outputs, ground_truth))
print("BERT F1:", bert_f1(llm_outputs, ground_truth))

print("\nBASIC RAG")
print("Pass Rate:", pass_rate(rag_outputs, ground_truth))
print("BERT F1:", bert_f1(rag_outputs, ground_truth))

print("\nGRAPH RAG")
print("Pass Rate:", pass_rate(graphrag_outputs, ground_truth))
print("BERT F1:", bert_f1(graphrag_outputs, ground_truth))
