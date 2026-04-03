import os
import tempfile
from pathlib import Path

import streamlit as st

from rag import (
    ask_question,
    build_index_from_documents,
    load_or_create_index,
)
from utils import (
    delete_file,
    read_docx,
    read_pdf,
    read_txt,
)

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.markdown(
    """
<style>
body {
    background: linear-gradient(135deg, #1e1e2f, #2b5876);
}
.block-container {
    padding-top: 1.5rem;
}
.stButton button {
    background: linear-gradient(45deg, #ff6ec4, #7873f5);
    color: white;
    border-radius: 10px;
    border: none;
}
.stTextInput input {
    border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("RAG QA System")
st.caption("Upload documents, build the database, and ask questions.")

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "faiss_metadata" not in st.session_state:
    st.session_state.faiss_metadata = None

if "session_mode" not in st.session_state:
    st.session_state.session_mode = "saved"


def extract_uploaded_documents(uploaded_files):
    docs = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            if suffix == ".txt":
                text = read_txt(tmp_path)
                if text.strip():
                    docs.append({"source": uploaded_file.name, "page": 1, "text": text})

            elif suffix == ".pdf":
                for page_no, text in read_pdf(tmp_path):
                    if text.strip():
                        docs.append({"source": uploaded_file.name, "page": page_no, "text": text})

            elif suffix == ".docx":
                text = read_docx(tmp_path)
                if text.strip():
                    docs.append({"source": uploaded_file.name, "page": 1, "text": text})

        finally:
            if tmp_path:
                delete_file(tmp_path)

    return docs


def load_backend_index():
    index, metadata = load_or_create_index()
    st.session_state.faiss_index = index
    st.session_state.faiss_metadata = metadata
    st.session_state.session_mode = "saved"


left, right = st.columns([1, 3])

with left:
    st.header("Upload documents")

    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    if st.button("Create Database", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            try:
                docs = extract_uploaded_documents(uploaded_files)
                if not docs:
                    st.error("No readable text found in the uploaded files.")
                else:
                    index, metadata = build_index_from_documents(
                        docs,
                        processed_dir="data/processed",
                        overwrite=True,
                        window_size=2,
                        overlap=0,
                    )
                    st.session_state.faiss_index = index
                    st.session_state.faiss_metadata = metadata
                    st.session_state.session_mode = "uploaded"
                    st.success("Database created successfully.")
            except Exception as e:
                st.error(f"Database creation failed: {e}")

    if st.button("Load Saved Database", use_container_width=True):
        try:
            load_backend_index()
            st.success("Saved database loaded successfully.")
        except Exception as e:
            st.warning(f"No saved database found: {e}")

with right:
    if os.path.exists("static/gi_logo.png"):
        st.image("static/gi_logo.png", width=500)
    else:
        st.warning("Logo not found at static/gi_logo.png")

    st.header("Search Your Documents")

    if st.session_state.faiss_index is not None:
        query = st.text_input("Enter query")

        if st.button("Search", use_container_width=True):
            if query.strip():
                answer, retrieved = ask_question(
                    st.session_state.faiss_index,
                    st.session_state.faiss_metadata,
                    query,
                )

                st.markdown("### Answer")
                st.success(answer)

                with st.expander("Retrieved context"):
                    if retrieved:
                        for item in retrieved:
                            st.write(item["text"])
                            st.write("---")
                    else:
                        st.write("No relevant chunks found.")
            else:
                st.warning("Please enter a query.")
    else:
        st.warning("Please upload documents or load a saved database first.")

if st.session_state.faiss_index is None:
    try:
        load_backend_index()
        st.rerun()
    except Exception:
        pass