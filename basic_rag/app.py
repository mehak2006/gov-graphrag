import streamlit as st
import json
import time

from rag_pipeline import (
    load_pdf,
    create_documents,
    create_embeddings,
    create_vectorstore,
    retrieve_documents,
    create_context,
    generate_answer
)

# PAGE CONFIG

st.set_page_config(
    page_title="Government Scheme Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Government Scheme Assistant")

st.write(
    "Ask questions about government schemes using AI-powered PDF retrieval."
)

# LOAD DATA

@st.cache_resource
def setup_rag_pipeline():

    pdf_path = "data/schemes.pdf"

    full_text = load_pdf(pdf_path)

    documents = create_documents(full_text)

    embeddings = create_embeddings()

    vectorstore = create_vectorstore(
        documents,
        embeddings
    )

    return vectorstore


vectorstore = setup_rag_pipeline()

# USER INPUT

query = st.text_input(
    "Enter your question:"
)

# PROCESS QUERY

if query:

    with st.spinner("Searching schemes..."):

        # Retrieval
        results, retrieval_time = retrieve_documents(
            vectorstore,
            query
        )

        # Context
        context = create_context(results)

        # LLM
        answer, llm_time, token_metrics = generate_answer(
            query,
            context
        )

        # Metrics
       
        metrics = {
 	   "retrieval_time_seconds": round(retrieval_time, 3),
    	   "llm_response_time_seconds": round(llm_time, 3),
    	   "total_latency_seconds": round(retrieval_time + llm_time, 3),
	   "chunks_retrieved": len(results),
    	   "context_length": len(context),
    	   "prompt_tokens": token_metrics["prompt_tokens"],
           "completion_tokens": token_metrics["completion_tokens"],
           "total_tokens": token_metrics["total_tokens"]
        }

    # FINAL ANSWER
    st.subheader("✅ Final Answer")

    st.write(answer)

    # RETRIEVED CHUNKS
    st.subheader("📄 Retrieved Scheme Chunks")

    for i, doc in enumerate(results):

         with st.expander(f"Chunk {i+1}"):

             st.write(doc.page_content)

    # METRICS
    st.subheader("📊 Metrics")

    st.json(metrics)
