import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

from utils import (
    split_sentences,
    create_chunks,
    load_local_documents,
    save_metadata,
    load_metadata,
    save_embeddings_readable,
)

EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
LLM_MODEL_NAME = "mistral"
TOP_K = 3

embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts, prefix):
    texts = [f"{prefix}: {t}" for t in texts]
    emb = embed_model.encode(texts)
    emb = np.array(emb, dtype="float32")
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb


def embed_chunks(chunks):
    if not chunks:
        raise ValueError("No chunks found. Add documents first.")
    return embed_texts(chunks, "passage")


def embed_query(query):
    return embed_texts([query], "query")


def create_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def reset_processed_files(processed_dir="data/processed"):
    index_path = os.path.join(processed_dir, "index.faiss")
    metadata_path = os.path.join(processed_dir, "metadata.json")

    for path in [index_path, metadata_path]:
        if os.path.exists(path):
            os.remove(path)


def build_index_from_documents(documents, processed_dir="data/processed", overwrite=True, window_size=2, overlap=0):
    all_chunks = []

    for doc in documents:
        sentences = split_sentences(doc["text"])
        if not sentences:
            continue

        doc_chunks = create_chunks(
            sentences=sentences,
            source=doc["source"],
            page=doc["page"],
            window_size=window_size,
            overlap=overlap,
        )
        all_chunks.extend(doc_chunks)

    if not all_chunks:
        raise ValueError("No chunks could be created from the provided documents.")

    for idx, chunk in enumerate(all_chunks):
        chunk["id"] = idx

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_chunks(texts)

    save_embeddings_readable(
        chunks=all_chunks,
        embeddings=embeddings,
        path=os.path.join(processed_dir, "embeddings.txt"),
    )

    index = create_index(embeddings)

    os.makedirs(processed_dir, exist_ok=True)
    if overwrite:
        reset_processed_files(processed_dir)

    faiss.write_index(index, os.path.join(processed_dir, "index.faiss"))
    save_metadata(all_chunks, os.path.join(processed_dir, "metadata.json"))

    return index, all_chunks


def load_index(processed_dir="data/processed"):
    index_path = os.path.join(processed_dir, "index.faiss")
    metadata_path = os.path.join(processed_dir, "metadata.json")

    index = faiss.read_index(index_path)
    metadata = load_metadata(metadata_path)

    if index.ntotal != len(metadata):
        raise ValueError("Index and metadata size mismatch.")

    return index, metadata


def load_or_create_index(data_dir="data", processed_dir="data/processed", window_size=2, overlap=0):
    index_path = os.path.join(processed_dir, "index.faiss")
    metadata_path = os.path.join(processed_dir, "metadata.json")

    try:
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            return load_index(processed_dir)
    except Exception:
        pass

    documents = load_local_documents(data_dir)
    if not documents:
        raise ValueError("No documents found. Upload files or add documents in data/.")

    return build_index_from_documents(
        documents=documents,
        processed_dir=processed_dir,
        overwrite=True,
        window_size=window_size,
        overlap=overlap,
    )


def retrieve(index, query_vec, metadata, query, k=TOP_K):
    if not metadata:
        return []

    top_k = min(k, len(metadata))
    sims, ids = index.search(query_vec, top_k)

    q_terms = [t for t in query.split() if t.strip()]
    scored = []

    for sim, idx in zip(sims[0], ids[0]):
        if idx < 0:
            continue

        item = metadata[idx]
        text = item["text"]
        overlap = sum(1 for term in q_terms if term in text)

        if overlap > 0:
            scored.append((overlap, float(sim), item))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    seen = set()
    out = []
    for _, _, item in scored:
        key = (item["source"], item["page"], item["position"], item["text"])
        if key not in seen:
            seen.add(key)
            out.append(item)

    return out


SYSTEM_PROMPT = """
You are a strict grounded QA system.

Rules:
- Answer only from the provided context.
- Answer in Hindi only.
- English words may appear only if they already appear in the context.
- Do not use prior knowledge.
- Do not guess.
- If the answer is not clearly present in context, output exactly: मुझे जानकारी नहीं है
- Do not explain anything.
- Return only the final answer.
""".strip()


def generate_answer(chunks, query):
    if not chunks:
        return "मुझे जानकारी नहीं है"

    context = "\n".join(c["text"] for c in chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer only from context."

    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0},
    )

    return response["message"]["content"].strip()


def ask_question(index, metadata, query):
    query_vec = embed_query(query)
    retrieved_chunks = retrieve(index, query_vec, metadata, query)
    answer = generate_answer(retrieved_chunks, query)
    return answer, retrieved_chunks


def build_index_from_local_folder():
    return load_or_create_index()


if __name__ == "__main__":
    index, metadata = load_or_create_index()

    while True:
        query = input("प्रश्न: ").strip()
        if query.lower() == "exit":
            break

        answer, retrieved = ask_question(index, metadata, query)

        print("\nRetrieved chunks:")
        for item in retrieved:
            print(item)

        print("\nAnswer:")
        print(answer)
        print("-" * 50)