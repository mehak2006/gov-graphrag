from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv
import os
import time
import json

# LOAD ENV VARIABLES

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# LOAD PDF

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    full_text = ""

    for doc in docs:
        full_text += doc.page_content + "\n"

    return full_text

# CREATE DOCUMENTS

def create_documents(full_text):
    schemes = full_text.split("========================")

    clean_schemes = []

    for scheme in schemes:
        scheme = scheme.strip()

        if len(scheme) > 50:
            clean_schemes.append(scheme)

    documents = []

    for scheme in clean_schemes:
        documents.append(
            Document(page_content=scheme)
        )

    return documents

# CREATE EMBEDDINGS

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu",
        }
    )

    return embeddings

# CREATE VECTOR STORE

def create_vectorstore(documents, embeddings):
    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    vectorstore.save_local("vector_store")

    return vectorstore

# RETRIEVE DOCUMENTS

def retrieve_documents(vectorstore, query, k=3):
    start_time = time.time()

    results = vectorstore.similarity_search(query, k=k)

    retrieval_time = time.time() - start_time

    return results, retrieval_time

# CREATE CONTEXT

def create_context(results):
    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    return context

# GENERATE ANSWER

def generate_answer(query, context):
    prompt = f"""
    You are a government scheme assistant.

    Answer ONLY from the provided context.

    If information is not available, say:
    "Insufficient information."

    Context:
    {context}

    Question:
    {query}
    """

    llm_start = time.time()

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant"
    )

    llm_time = time.time() - llm_start

    answer = chat_completion.choices[0].message.content

    usage = chat_completion.usage

    metrics = {
    "prompt_tokens": usage.prompt_tokens,
    "completion_tokens": usage.completion_tokens,
    "total_tokens": usage.total_tokens
    }

    return answer, llm_time, metrics
