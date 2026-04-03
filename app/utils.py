import os
import re
from pypdf import PdfReader
from docx import Document


def split_sentences(text):
    parts = re.split(r"[।.!?]\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(path):
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((i + 1, text))

    return pages


def read_docx(path):
    doc = Document(path)
    return [(1, "\n".join(p.text for p in doc.paragraphs))]


def load_documents(folder):
    docs = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".txt"):
            docs.append((file, 1, read_txt(path)))

        elif file.endswith(".pdf"):
            pages = read_pdf(path)
            for page_no, text in pages:
                docs.append((file, page_no, text))

        elif file.endswith(".docx"):
            docs.extend([(file, 1, t[1]) for t in read_docx(path)])

    return docs


def create_chunks(text, source, page, window=2):
    sentences = split_sentences(text)
    chunks = []

    for i in range(0, len(sentences), window):
        chunk_text = " ".join(sentences[i:i + window])

        chunks.append({
            "id": len(chunks),
            "text": chunk_text,
            "source": source,
            "page": page,
            "position": i
        })

    return chunks


def save_metadata(data, path="data/processed/metadata.json"):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_metadata(path="data/processed/metadata.json"):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)