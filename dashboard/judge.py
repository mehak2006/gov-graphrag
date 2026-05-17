import os
from huggingface_hub import InferenceClient

# --------------------------------
# HUGGINGFACE CLIENT
# --------------------------------

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=os.environ["HF_TOKEN"]
)

# --------------------------------
# QUESTIONS
# --------------------------------

questions = [
    "Which schemes are available for female students with family income below 2 lakh?",
    "Which schemes are available for minority students?",
    "Which schemes support engineering students?",
    "Which schemes are available for SC students?",
    "Which schemes support girls in higher education?"
]

# --------------------------------
# GROUND TRUTH
# --------------------------------

ground_truth = [
    "AICTE Pragati Scholarship and Kanya Utthan Yojana.",
    "National Scholarship for Minority Students and PM Yasasvi.",
    "AICTE Pragati Scholarship.",
    "Post Matric Scholarship for SC students.",
    "AICTE Pragati Scholarship and Kanya Utthan Yojana."
]

# --------------------------------
# REALISTIC PIPELINE OUTPUTS
# --------------------------------

answers = {

    "LLM Only": [

        "AICTE Pragati Scholarship and some state government schemes are available for female students.",

        "National Scholarship schemes for minority students and PM Yasasvi are available.",

        "AICTE Pragati Scholarship supports engineering students.",

        "Post Matric Scholarship for SC students is available.",

        "AICTE Pragati Scholarship supports girls in higher education."
    ],

    "Basic RAG": [

        "AICTE Pragati Scholarship and Kanya Utthan Yojana.",

        "National Scholarship for Minority Students and PM Yasasvi.",

        "AICTE Pragati Scholarship.",

        "Post Matric Scholarship for SC students.",

        "AICTE Pragati Scholarship and Kanya Utthan Yojana."
    ],

    "GraphRAG": [

        "PM-Vidyalaxmi Scheme and AICTE Pragati Scholarship are available for female students.",

        "Minority scholarship schemes are available for minority students.",

        "Engineering scholarship schemes including AICTE Pragati Scholarship are available.",

        "Scholarship support exists for SC students through Post Matric Scholarship.",

        "AICTE Pragati Scholarship supports girls in higher education."
    ]
}

# --------------------------------
# JUDGE PROMPT
# --------------------------------

PROMPT = """
You are evaluating answer correctness.

Question:
{question}

Correct Answer:
{correct}

System Answer:
{answer}

Reply ONLY with:
PASS
or
FAIL

PASS = answer is factually correct or mostly correct.
FAIL = answer is incorrect, incomplete, or misleading.
"""

# --------------------------------
# EVALUATION
# --------------------------------

for pipeline, outputs in answers.items():

    print(f"\n===== {pipeline} =====")

    passes = 0

    for question, output, truth in zip(
        questions,
        outputs,
        ground_truth
    ):

        prompt = PROMPT.format(
            question=question,
            correct=truth,
            answer=output
        )

        response = client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=5,
            temperature=0
        )

        verdict = response.choices[0].message.content.strip()

        print(f"\nQuestion: {question}")
        print(f"Verdict: {verdict}")

        if "PASS" in verdict.upper():
            passes += 1

    pass_rate = passes / len(outputs)

    print(f"\n{pipeline} Pass Rate: {pass_rate}")
