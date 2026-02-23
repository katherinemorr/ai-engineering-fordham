# streamlit run fordham_rag_app.py
# streamlit run fordham_rag_app.py

#temp_dir = Path("data")
# Load embeddings 
#loaded_embeddings = np.load(temp_dir / "website_embeddings.npy")
#loaded_embeddings = np.load("/Users/kmorris22/ce-personal/temp/website_embeddings.npy")

# Load chunks 
#loaded_chunks = pd.read_csv(temp_dir / "chunks_sample.csv")
#loaded_chunks = pd.read_csv("/Users/kmorris22/ce-personal/temp/chunks_sample.csv")


import streamlit as st
import numpy as np
import pandas as pd
import litellm
from pathlib import Path

# Load environment variables for API keys
from dotenv import load_dotenv
load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fordham University Q&A",
    page_icon="🐏",
    layout="centered",
)

# ── Load data (cached so it only runs once) ───────────────────────────────────
@st.cache_resource
def load_data():
    temp_dir = Path("temp")
    embeddings = np.load(temp_dir / "website_embeddings.npy")
    chunks = pd.read_csv(temp_dir / "chunks_sample.csv")
    return embeddings, chunks

loaded_embeddings, loaded_chunks = load_data()

# ── Helper functions ──────────────────────────────────────────────────────────
def batch_cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between query and all documents."""
    query_norm = query_emb / np.linalg.norm(query_emb)
    doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    return doc_norms @ query_norm


def rag(query: str) -> tuple[str, list[str]]:
    """
    Run the RAG pipeline for a given query.

    Returns:
        answer: LLM-generated answer string
        source_urls: list of URLs used as sources
    """
    k = 5

    # Embed query
    emb_response = litellm.embedding(model="text-embedding-3-small", input=[query])
    query_emb = np.array(emb_response.data[0]["embedding"])

    # Retrieve top-k chunks
    similarities = batch_cosine_similarity(query_emb, loaded_embeddings)
    top_k_idx = np.argsort(-similarities)[:k]
    search_results = loaded_chunks.iloc[top_k_idx].copy()
    search_results["similarity"] = similarities[top_k_idx]

    retrieved_texts = search_results["content"].tolist()
    retrieved_urls = search_results["url"].tolist()

    # Build context
    context_items = []
    for i, (content, url) in enumerate(zip(retrieved_texts, retrieved_urls)):
        context_items.append(f"Source {i+1} [URL: {url}]:\n{content}")
    context_text = "\n\n".join(context_items)

    # Prompts
    system_prompt = """You are a helpful and accurate assistant for Fordham University.
Your task is to answer the user's question using ONLY the provided context.

- Summarize the answer clearly based on the context.
- If the answer is NOT in the context, say: "I'm sorry, I don't have that information."
- At the end of your response, provide a section titled "Sources:" followed by a
  numbered list of the URLs you used to form the answer."""

    user_prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""

    # Call LLM
    llm_response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = llm_response.choices[0].message.content
    return answer, retrieved_urls


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐏 Fordham University Q&A")
st.caption("Ask anything about Fordham — powered by RAG.")

query = st.text_input(
    "Your question",
    placeholder="e.g. What are the application deadlines for Fordham?",
)

if st.button("Ask", type="primary", disabled=not query):
    with st.spinner("Searching and generating answer…"):
        try:
            answer, source_urls = rag(query)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # ── Answer ──
    st.subheader("Answer")
    # Strip the Sources section from the answer body if the LLM included it,
    # since we display sources separately below.
    answer_body = answer.split("Sources:")[0].strip()
    st.markdown(answer_body)

    # ── Sources ──
    unique_urls = list(dict.fromkeys(source_urls))  # preserve order, deduplicate
    st.subheader("Source Pages Used")
    for i, url in enumerate(unique_urls, 1):
        st.markdown(f"{i}. [{url}]({url})")