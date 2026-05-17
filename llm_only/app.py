import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import time

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

st.title("LLM Only Pipeline")

query = st.text_input("Ask a question")

if query:

    start = time.time()

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query
            }
        ],
        model="llama-3.1-8b-instant"
    )

    latency = time.time() - start

    answer = response.choices[0].message.content

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Metrics")
    st.json({
        "latency_seconds": round(latency, 3),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    })
