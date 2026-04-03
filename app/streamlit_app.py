import streamlit as st
from rag import load_or_create, ask

st.set_page_config(page_title="RAG", layout="wide")

st.title("Hybrid RAG System")

index, metadata, bm25 = load_or_create()

query = st.text_input("Ask question")

if st.button("Get Answer"):
    ans, chunks = ask(index, metadata, bm25, query)

    st.success(ans)

    with st.expander("Metadata"):
        for c in chunks:
            st.write(f"Source: {c['source']}")
            st.write(f"Page: {c['page']}")
            st.write(f"Position: {c['position']}")
            st.write(f"Text: {c['text']}")
            st.write("---")