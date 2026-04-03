import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from rank_bm25 import BM25Okapi

from utils import load_documents, create_chunks, save_metadata, load_metadata

MODEL = SentenceTransformer("intfloat/multilingual-e5-base")
LLM = "mistral"


def embed(texts, prefix):
    texts = [f"{prefix}: {t}" for t in texts]
    emb = MODEL.encode(texts)
    emb = np.array(emb).astype("float32")
    faiss.normalize_L2(emb)
    return emb


def build_index():
    docs = load_documents("data")

    all_chunks = []

    for source, page, text in docs:
        chunks = create_chunks(text, source, page)
        all_chunks.extend(chunks)

    texts = [c["text"] for c in all_chunks]

    embeddings = embed(texts, "passage")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # BM25
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    os.makedirs("data/processed", exist_ok=True)
    faiss.write_index(index, "data/processed/index.faiss")
    save_metadata(all_chunks)

    return index, all_chunks, bm25


def load_index():
    index = faiss.read_index("data/processed/index.faiss")
    metadata = load_metadata()

    texts = [c["text"] for c in metadata]
    bm25 = BM25Okapi([t.split() for t in texts])

    return index, metadata, bm25


def load_or_create():
    if os.path.exists("data/processed/index.faiss"):
        return load_index()
    return build_index()


def hybrid_search(query, index, metadata, bm25):
    q_vec = embed([query], "query")

    # FAISS
    sims, ids = index.search(q_vec, 5)

    # BM25
    scores = bm25.get_scores(query.split())

    results = []

    for i in range(len(metadata)):
        faiss_score = 0
        if i in ids[0]:
            faiss_score = sims[0][list(ids[0]).index(i)]

        score = faiss_score + scores[i]

        results.append((score, metadata[i]))

    results.sort(reverse=True, key=lambda x: x[0])

    return [r[1] for r in results[:3]]


def generate_answer(chunks, query):
    context = "\n".join(c["text"] for c in chunks)

    prompt = f"""
Context:
{context}

Question:
{query}

Answer only from context in Hindi.
If not present say: मुझे जानकारी नहीं है
"""

    res = ollama.chat(
        model=LLM,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return res["message"]["content"]


def ask(index, metadata, bm25, query):
    chunks = hybrid_search(query, index, metadata, bm25)
    answer = generate_answer(chunks, query)
    return answer, chunks


if __name__ == "__main__":
    index, metadata, bm25 = load_or_create()

    while True:
        q = input("प्रश्न: ")
        if q == "exit":
            break

        ans, ch = ask(index, metadata, bm25, q)

        print("\nChunks:")
        for c in ch:
            print(c)

        print("\nAnswer:", ans)