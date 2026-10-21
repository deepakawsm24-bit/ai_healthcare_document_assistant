import re
import hashlib
from collections import Counter

import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="AI Healthcare Document Assistant",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 AI Healthcare Document Assistant")
st.caption("Ask questions about uploaded PDFs. Answers are grounded in retrieved document text.")

st.warning(
    "For learning and demonstration only. Do not upload real patient-identifiable records "
    "or confidential hospital documents. This app does not diagnose conditions or recommend treatment."
)

STOP_WORDS = {
    "what", "how", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an",
    "and", "or", "of", "for", "in", "on", "to", "it", "its", "by", "with", "from",
    "this", "that", "these", "those", "be", "as", "at", "about", "according", "uploaded",
    "pdf", "pdfs", "document", "documents", "please", "tell", "me", "give", "show",
    "main", "difference", "differences", "between", "compare", "comparison", "compared",
    "explain", "describe", "summarize", "summary", "list", "based", "your",
}

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def tokenize(text):
    # Keep technical terms such as CNN, RNN, ResNet, and MobileNet.
    return re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", text.lower())


def query_terms(text):
    return {term for term in tokenize(text) if term not in STOP_WORDS and len(term) > 1}


def extract_pdf_text(uploaded_file):
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    page_texts = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        if text.strip():
            page_texts.append((page_no, text.strip()))
    return page_texts, len(reader.pages)


def split_into_passages(page_texts, source_name, max_chars=1100, overlap=140):
    passages = []
    for page_no, page_text in page_texts:
        paragraphs = [
            re.sub(r"\s+", " ", p).strip()
            for p in re.split(r"\n\s*\n|\r\n\s*\r\n", page_text)
            if p.strip()
        ]
        if not paragraphs:
            paragraphs = [re.sub(r"\s+", " ", page_text).strip()]

        current = ""
        for paragraph in paragraphs:
            if len(paragraph) > max_chars:
                if current:
                    passages.append({
                        "source": source_name, "page": page_no, "text": current
                    })
                    current = ""
                start = 0
                while start < len(paragraph):
                    piece = paragraph[start:start + max_chars].strip()
                    if piece:
                        passages.append({
                            "source": source_name, "page": page_no, "text": piece
                        })
                    if start + max_chars >= len(paragraph):
                        break
                    start += max_chars - overlap
            elif not current:
                current = paragraph
            elif len(current) + len(paragraph) + 2 <= max_chars:
                current += "\n\n" + paragraph
            else:
                passages.append({
                    "source": source_name, "page": page_no, "text": current
                })
