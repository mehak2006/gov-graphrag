# 📚 YojanaGraph — Benchmarking LLM-Only vs Basic RAG vs GraphRAG for Government Scheme Retrieval

YojanaGraph is a benchmarking project built for the TigerGraph GraphRAG Inference Hackathon.

The project compares three different retrieval architectures side-by-side on Indian Government Scheme retrieval tasks:

- LLM-Only
- Basic RAG
- GraphRAG using TigerGraph

The goal was to evaluate how retrieval architecture affects:

- factual grounding
- hallucination reduction
- token usage
- latency
- semantic answer quality

---

# 🚀 Live Demo

## Public Streamlit Deployment

The deployed Streamlit version currently demonstrates:

- LLM-Only pipeline
- Basic RAG pipeline
- GraphRAG benchmark outputs + metrics

Due to infrastructure constraints explained below, the full live TigerGraph backend is not publicly hosted in the deployed version.

---

# ☁️ Public Deployment vs 🖥️ Full Local MVP

This repository contains two separate dashboard configurations:

## 1. Cloud Deployment (`dashboard/app.py`)

The publicly deployed Streamlit version is a lightweight cloud-safe dashboard intended for:
- benchmark visualization
- architecture demonstration
- retrieval comparison
- cached GraphRAG outputs

This version avoids heavyweight infrastructure dependencies that are unsupported on Streamlit Community Cloud, such as:
- Docker containers
- TigerGraph server hosting
- FAISS indexing
- local embedding pipelines

Run locally:

```bash
streamlit run dashboard/app.py
```

Install lightweight dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Full Local GraphRAG MVP (`dashboard/app_local.py`)

The complete working MVP is available locally and includes:
- live TigerGraph GraphRAG retrieval
- FAISS vector search
- HuggingFace embeddings
- Docker-based TigerGraph services
- real-time graph querying
- graph ingestion + entity extraction

This version represents the actual end-to-end GraphRAG system used during experimentation and benchmarking.

Run locally:

```bash
streamlit run dashboard/app_local.py
```

Install full dependencies:

```bash
pip install -r requirements-local.txt
```

---

# 🕸️ Why the Full GraphRAG System Is Not Publicly Hosted

The GraphRAG backend depends on:
- TigerGraph database services
- Docker infrastructure
- local graph APIs
- persistent graph storage

These components are not supported on Streamlit Community Cloud.

As a result:
- the public deployment demonstrates benchmark outputs and architecture
- the fully functional GraphRAG MVP is intended for local execution

The repository includes:
- source code
- local setup instructions
- benchmark scripts
- screenshots
- evaluation results
- GraphRAG integration code

---

# 🕸️ GraphRAG Deployment Note

The GraphRAG pipeline depends on:

- TigerGraph database services
- Docker containers
- local REST APIs
- graph ingestion infrastructure

The current GraphRAG implementation is an MVP prototype running locally.

While the Streamlit dashboard is publicly deployed, the complete GraphRAG backend was not deployed publicly because Streamlit Community Cloud does not support:

- Docker-based infrastructure
- persistent graph databases
- TigerGraph server hosting
- localhost REST API access

As a result:

- the public deployment contains cached/demo GraphRAG outputs and benchmark metrics
- the fully functional GraphRAG system is demonstrated through:
  - local demo videos
  - screenshots
  - benchmark reports
  - architecture diagrams

This architecture separation reflects a realistic production setup where:
- frontend dashboards are hosted separately
- graph databases run on dedicated infrastructure/VMs

---

# 🧠 Pipelines

## 1. LLM-Only

The baseline pipeline directly queries the LLM without retrieval augmentation.

Observed behavior:

- highest hallucination risk
- verbose answers
- highest token usage
- weakest factual grounding

---

## 2. Basic RAG

The Basic RAG pipeline uses:

- LangChain
- HuggingFace embeddings
- FAISS vector search
- Groq-hosted LLM inference

Pipeline flow:

1. Chunk documents
2. Create embeddings
3. Retrieve semantically relevant chunks
4. Inject retrieved context into prompt
5. Generate grounded answer

This pipeline achieved the strongest factual performance during evaluation.

---

## 3. GraphRAG (TigerGraph)

The GraphRAG pipeline attempts to:

- construct a knowledge graph
- extract entities and relationships
- retrieve graph-grounded context

The objective is to improve:

- relationship-aware retrieval
- multi-hop reasoning
- structured grounding
- eligibility-based filtering

The project uses TigerGraph for:

- graph storage
- graph ingestion
- retrieval APIs
- GraphRAG workflows

---

# 📊 Benchmark Results

- **LLM-Only:** Avg latency = 1.42s · Avg tokens/query = 597.5 · LLM-as-a-Judge pass rate = 0.80 · BERTScore F1 = 0.875

- **Basic RAG:** Avg latency = 0.57s · Avg tokens/query = 229.5 · LLM-as-a-Judge pass rate = 1.00 · BERTScore F1 = 1.000

- **GraphRAG:** Avg latency = 0.535s · Avg tokens/query = 281.5 · LLM-as-a-Judge pass rate = 0.80 · BERTScore F1 = 0.876

- **Token Reduction (GraphRAG vs Basic RAG):** -22.7%

- **Estimated Cost per Query (GPT-4o-mini equivalent pricing):**
  - LLM-Only ≈ $0.0012
  - Basic RAG ≈ $0.0005
  - GraphRAG ≈ $0.0006

---

# 🔍 Key Observations

## Basic RAG achieved the strongest overall performance

For this dataset, semantic vector retrieval provided:

- strong factual grounding
- low hallucination rates
- high semantic similarity
- efficient token usage

---

## GraphRAG performance depended heavily on graph quality

One of the most interesting findings was that GraphRAG performance was highly dependent on:

- entity extraction quality
- relationship extraction quality
- completeness of graph ingestion

Incomplete graph construction reduced retrieval coverage and affected answer completeness.

This created a precision–recall tradeoff:

- GraphRAG produced focused answers
- but sometimes missed broader relevant schemes

---

## LLM-Only produced the highest hallucination risk

Without retrieval grounding, the model frequently generated:

- plausible but unsupported schemes
- verbose outputs
- inconsistent eligibility information

---
## ⚠️ GraphRAG Performance Note

The current GraphRAG benchmark should be interpreted as an MVP prototype evaluation rather than a fully optimized production GraphRAG deployment.

During experimentation, the local TigerGraph GraphRAG application consistently produced stronger responses than the lightweight custom dashboard integration. This difference was primarily due to infrastructure and inference constraints, including:

- Groq free-tier token-per-minute rate limits during graph extraction
- long-running graph rebuild times on large corpora
- partial graph construction during intermediate evaluations
- simplified API integration inside the custom dashboard
- reduced retrieval orchestration compared to the native TigerGraph GraphRAG UI

As a result, the full capabilities of TigerGraph GraphRAG were not completely exploited in the benchmarked dashboard environment.

The native localhost GraphRAG application demonstrated:
- richer graph-grounded retrieval
- better multi-hop reasoning
- improved entity-aware responses
- more accurate eligibility-based retrieval

This project therefore reflects:
- a realistic resource-constrained MVP GraphRAG deployment
- early-stage benchmarking under limited inference budgets
- practical infrastructure tradeoffs between vector RAG and graph RAG systems

Future iterations with:
- higher inference throughput
- optimized graph extraction
- stronger retrieval orchestration
- dedicated backend infrastructure

would likely improve GraphRAG performance substantially.
---


# 🛠️ Tech Stack

- TigerGraph
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq API
- Python

---

# 📂 Project Structure

```bash
gov-graphrag/
│
├── dashboard/
│   ├── app.py
│   └── app_local.py
│
├── basic_rag/
├── llm_only/
├── graphrag/
├── data/
├── benchmark_questions.json
├── evaluate.py
├── judge.py
├── requirements.txt
├── requirements-local.txt
└── README.md
```

---

# 📸 Demo Assets

The repository includes:

- local GraphRAG screenshots
- benchmark outputs
- evaluation scripts
- architecture diagrams
- demo videos

These demonstrate the fully functional local GraphRAG prototype.

---

# 📌 Future Improvements

Potential future improvements include:

- improved graph extraction pipelines
- richer entity linking
- larger evaluation benchmarks
- public VM deployment for TigerGraph backend
- production-grade GraphRAG orchestration

---

# 💡 Final Takeaway

This project highlighted an important insight:

> Retrieval quality fundamentally shapes LLM behavior.

GraphRAG is extremely promising, but its performance depends heavily on reliable graph construction pipelines.

Even in this MVP prototype, the differences between:

- semantic retrieval
- graph retrieval
- no retrieval

became immediately visible through benchmarking.

The project ultimately became less about prompt engineering and more about understanding how retrieval architecture influences reasoning, grounding, and factual reliability in modern LLM systems.