import json
import os
import re
from pypdf import PdfReader
from docx import Document


def split_sentences(text):
    text = text.replace("\r", "\n").strip()
    if not text:
        return []
    parts = re.split(r"[।.!?]\s+|\n+", text)
    return [p.strip(" \t\n\r।.!?") for p in parts if p.strip(" \t\n\r।.!?")]


def create_chunks(sentences, source, page, window_size=2, overlap=0):
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size")

    chunks = []
    step = window_size - overlap

    for i in range(0, len(sentences), step):
        window = sentences[i:i + window_size]
        if not window:
            continue

        chunk_text = " ".join(window).strip()
        if not chunk_text:
            continue

        chunks.append({
            "id": len(chunks),
            "text": chunk_text,
            "source": source,
            "page": page,
            "position": i
        })

    return chunks


def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(path):
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))

    return pages


def read_docx(path):
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return text


def load_local_documents(folder):
    docs = []

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue

        lower = name.lower()
        if lower.endswith(".txt"):
            text = read_txt(path)
            if text.strip():
                docs.append({"source": name, "page": 1, "text": text})

        elif lower.endswith(".pdf"):
            for page_no, text in read_pdf(path):
                if text.strip():
                    docs.append({"source": name, "page": page_no, "text": text})

        elif lower.endswith(".docx"):
            text = read_docx(path)
            if text.strip():
                docs.append({"source": name, "page": 1, "text": text})

    return docs


def save_metadata(chunks, path="data/processed/metadata.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_metadata(path="data/processed/metadata.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_embeddings_readable(chunks, embeddings, path="data/processed/embeddings.txt"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            f.write(f"Chunk ID: {chunk['id']}\n")
            f.write(f"Source: {chunk['source']}\n")
            f.write(f"Page: {chunk['page']}\n")
            f.write(f"Position: {chunk['position']}\n")
            f.write(f"Text: {chunk['text']}\n")

            f.write("Embedding: [")
            f.write(", ".join(f"{x:.6f}" for x in vec))
            f.write("]\n")

            f.write("-" * 80 + "\n")

def delete_file(path):
    if path and os.path.exists(path):
        os.remove(path)
