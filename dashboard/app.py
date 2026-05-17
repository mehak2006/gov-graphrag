import streamlit as st
import time
import json
import os
import sys
import requests
from groq import Groq
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------
# LOAD ENV
# -----------------------------------

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------------
# IMPORT BASIC RAG (only what we need)
# -----------------------------------

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "basic_rag"))

from rag_pipeline import (
    create_embeddings,
    create_vectorstore,
    retrieve_documents,
    create_context,
    generate_answer
)

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="GraphRAG Benchmark Dashboard",
    layout="wide"
)

st.title("GraphRAG vs Basic RAG vs LLM Benchmark")
st.write("Real-time comparison of LLM-only, Basic RAG, and GraphRAG pipelines on Indian Government Schemes dataset.")

# -----------------------------------
# LOAD VECTORSTORE FROM JSON DATASET
# -----------------------------------

@st.cache_resource
def setup_rag():
    json_path = "/home/mehak2006/gov-graphrag/data/govt_schemes_large.json"

    with open(json_path) as f:
        data = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    documents = []
    for doc in data['documents']:
        text = f"Title: {doc['title']}\n\n{doc['content']}"
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk))

    st.sidebar.success(f"✅ Loaded {len(data['documents'])} articles → {len(documents)} chunks")

    embeddings = create_embeddings()
    vectorstore = create_vectorstore(documents, embeddings)
    return vectorstore

with st.spinner("Loading dataset into vector store..."):
    vectorstore = setup_rag()

# -----------------------------------
# HELPER: FETCH GRAPH CONTEXT
# -----------------------------------

def fetch_graph_context(query, top_k=2):
    """
    Fetch relevant chunks from TigerGraph knowledge graph.
    Falls back to fetching Content vertices directly.
    """
    # Try vector search first
    try:
        resp = requests.post(
            "http://localhost:8000/scheme_eligibilty_req/graphrag/search",
            json={"question": query, "method": "hybrid", "top_k": top_k},
            auth=("tigergraph", "tigergraph"),
            timeout=10
        )
        if resp.status_code == 200:
            results = resp.json()
            if isinstance(results, list) and len(results) > 0:
                return results, "hybrid_search"
    except Exception:
        pass

    # Fallback: fetch Content vertices directly from TigerGraph
    try:
        resp = requests.get(
            "http://localhost:14240/restpp/graph/scheme_eligibilty_req/vertices/Content",
            params={"limit": top_k},
            auth=("tigergraph", "tigergraph"),
            timeout=10
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            chunks = [r["attributes"].get("text", "") for r in results]
            return chunks, "direct_fetch"
    except Exception:
        pass

    return [], "none"

# -----------------------------------
# SIDEBAR INFO
# -----------------------------------

st.sidebar.title("Pipeline Info")
st.sidebar.markdown("""
**Pipeline 1 — LLM Only**
No retrieval. Direct query to LLM.

**Pipeline 2 — Basic RAG**
Vector similarity search over chunked Wikipedia articles → LLM.

**Pipeline 3 — GraphRAG**
TigerGraph knowledge graph retrieval → focused context → LLM.
""")

# -----------------------------------
# QUERY INPUT
# -----------------------------------

query = st.text_input("Enter Query", placeholder="e.g. Which schemes are available for SC female students?")

# -----------------------------------
# MAIN PIPELINES
# -----------------------------------

if query:

    col1, col2, col3 = st.columns(3)

    # ===================================
    # PIPELINE 1 — LLM ONLY
    # ===================================

    with st.spinner("Running LLM Only..."):
        llm_start = time.time()

        llm_response = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],

            model="llama-3.1-8b-instant"
        )

        llm_time = time.time() - llm_start
        llm_answer = llm_response.choices[0].message.content

    # ===================================
    # PIPELINE 2 — BASIC RAG
    # ===================================

    with st.spinner("Running Basic RAG..."):
        results, retrieval_time = retrieve_documents(vectorstore, query)
        context = create_context(results)
        rag_answer, rag_llm_time, rag_token_metrics = generate_answer(query, context)
        rag_total_time = retrieval_time + rag_llm_time

    # ===================================
    # PIPELINE 3 — GRAPH RAG
    # ===================================

    with st.spinner("Running GraphRAG (TigerGraph)..."):
        graph_start = time.time()

        try:
            graph_chunks, fetch_method = fetch_graph_context(query, top_k=10)

            if isinstance(graph_chunks, list) and len(graph_chunks) > 0:
                if isinstance(graph_chunks[0], str):
                    graph_context = "\n\n".join(graph_chunks)
                else:
                    graph_context = "\n\n".join([
                        str(c.get("text", c.get("content", str(c))))
                        for c in graph_chunks
                    ])
            else:
                graph_context = ""

            if not graph_context.strip():
                raise ValueError("No graph context retrieved")

            graph_prompt = f"""You are a helpful assistant with access to a knowledge graph about Indian government schemes and scholarships.

Use ONLY the following information retrieved from the knowledge graph to answer the question.

Knowledge Graph Context:
{graph_context[:800]}

Question: {query}

Answer based only on the knowledge graph context above. Be concise."""

            graph_llm_response = client.chat.completions.create(
                messages=[{"role": "user", "content": graph_prompt}],
                model="llama-3.1-8b-instant"
            )

            graph_time = time.time() - graph_start
            graph_answer = graph_llm_response.choices[0].message.content
            graph_tokens = graph_llm_response.usage
            graph_answered = True
            graph_chunks_used = len(graph_chunks)

        except Exception as e:
            graph_time = time.time() - graph_start
            graph_answer = f"GraphRAG Error: {str(e)}"
            graph_tokens = None
            graph_answered = False
            graph_chunks_used = 0
            fetch_method = "error"

    # ===================================
    # DISPLAY — LLM ONLY
    # ===================================

    with col1:
        st.subheader("🤖 LLM Only")
        st.write(llm_answer)
        st.subheader("Metrics")
        st.json({
            "latency_seconds": round(llm_time, 2),
            "prompt_tokens": llm_response.usage.prompt_tokens,
            "completion_tokens": llm_response.usage.completion_tokens,
            "total_tokens": llm_response.usage.total_tokens
        })

    # ===================================
    # DISPLAY — BASIC RAG
    # ===================================

    with col2:
        st.subheader("📚 Basic RAG")
        st.write(rag_answer)
        st.subheader("Metrics")
        st.json({
            "retrieval_time_seconds": round(retrieval_time, 2),
            "llm_time_seconds": round(rag_llm_time, 2),
            "total_latency_seconds": round(rag_total_time, 2),
            "chunks_retrieved": len(results),
            "context_length_chars": len(context),
            "prompt_tokens": rag_token_metrics["prompt_tokens"],
            "completion_tokens": rag_token_metrics["completion_tokens"],
            "total_tokens": rag_token_metrics["total_tokens"]
        })

    # ===================================
    # DISPLAY — GRAPH RAG
    # ===================================

    with col3:
        st.subheader("🕸️ GraphRAG (TigerGraph)")
        st.write(graph_answer)
        st.subheader("Metrics")
        st.json({
            "latency_seconds": round(graph_time, 2),
            "graph_chunks_used": graph_chunks_used,
            "fetch_method": fetch_method,
            "prompt_tokens": graph_tokens.prompt_tokens if graph_tokens else "N/A",
            "completion_tokens": graph_tokens.completion_tokens if graph_tokens else "N/A",
            "total_tokens": graph_tokens.total_tokens if graph_tokens else "N/A",
            "answered_question": graph_answered
        })

    # ===================================
    # TOKEN COMPARISON SUMMARY
    # ===================================

    st.divider()
    st.subheader("📊 Token Usage Comparison")

    llm_tokens = llm_response.usage.total_tokens
    rag_tokens = rag_token_metrics["total_tokens"]
    graph_total_tokens = graph_tokens.total_tokens if graph_tokens else 0

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric(
            label="🤖 LLM Only",
            value=f"{llm_tokens} tokens"
        )

    with summary_col2:
        rag_delta = rag_tokens - llm_tokens
        st.metric(
            label="📚 Basic RAG",
            value=f"{rag_tokens} tokens",
            delta=f"{rag_delta:+d} vs LLM Only",
            delta_color="inverse"
        )

    with summary_col3:
        if graph_total_tokens > 0:
            graph_delta = graph_total_tokens - rag_tokens
            graph_reduction = round((1 - graph_total_tokens / rag_tokens) * 100, 1)
            st.metric(
                label="🕸️ GraphRAG",
                value=f"{graph_total_tokens} tokens",
                delta=f"{graph_delta:+d} vs Basic RAG",
                delta_color="inverse"
            )
            if graph_reduction > 0:
                st.success(f"✅ GraphRAG used {graph_reduction}% fewer tokens than Basic RAG!")
            else:
                st.info(f"ℹ️ GraphRAG used {abs(graph_reduction)}% more tokens than Basic RAG.")
        else:
            st.metric(label="🕸️ GraphRAG", value="N/A")

    # ===================================
    # LATENCY COMPARISON
    # ===================================

    st.subheader("⚡ Latency Comparison")

    lat_col1, lat_col2, lat_col3 = st.columns(3)

    with lat_col1:
        st.metric("🤖 LLM Only", f"{round(llm_time, 2)}s")

    with lat_col2:
        st.metric("📚 Basic RAG", f"{round(rag_total_time, 2)}s")

    with lat_col3:
        st.metric("🕸️ GraphRAG", f"{round(graph_time, 2)}s")
