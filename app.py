import re
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
st.caption("Upload a PDF and ask questions about its contents.")

st.warning(
    "For learning and demonstration only. Do not upload real patient-identifiable "
    "records or confidential hospital documents. This app does not diagnose conditions "
    "or recommend treatment."
)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    page_texts = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            page_texts.append((page_no, text))
    return page_texts, len(reader.pages)

def split_into_passages(page_texts, max_chars=1000):
    passages = []
    for page_no, page_text in page_texts:
        # Keep paragraphs together when possible, then split long paragraphs.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        if not paragraphs:
            paragraphs = [page_text.strip()] if page_text.strip() else []
        current = ""
        for paragraph in paragraphs:
            if len(paragraph) > max_chars:
                if current:
                    passages.append({"page": page_no, "text": current})
                    current = ""
                for start in range(0, len(paragraph), max_chars - 120):
                    piece = paragraph[start:start + max_chars].strip()
                    if piece:
                        passages.append({"page": page_no, "text": piece})
            elif not current:
                current = paragraph
            elif len(current) + len(paragraph) + 2 <= max_chars:
                current += "\n\n" + paragraph
            else:
                passages.append({"page": page_no, "text": current})
                current = paragraph
        if current:
            passages.append({"page": page_no, "text": current})
    return passages

def tokenize(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

def build_index(passages, embedding_model):
    texts = [p["text"] for p in passages]
    vectors = embedding_model.encode(
        texts, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def retrieve_passages(question, passages, embedding_model, index, top_k=4):
    q_vector = embedding_model.encode(
        [question], convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    faiss.normalize_L2(q_vector)
    k = min(top_k, len(passages))
    scores, ids = index.search(q_vector, k)
    q_terms = tokenize(question) - {
        "what", "how", "does", "is", "the", "a", "an", "and", "of", "for",
        "in", "on", "to", "it", "main", "advantage", "work", "explain"
    }
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        passage = passages[int(idx)]
        p_terms = tokenize(passage["text"])
        overlap = len(q_terms & p_terms) / max(1, len(q_terms))
        # Semantic similarity plus a small exact-keyword-match boost.
        combined_score = float(score) + 0.20 * overlap
        results.append({
            "page": passage["page"],
            "text": passage["text"],
            "semantic_score": float(score),
            "combined_score": combined_score,
            "overlap": overlap,
        })
    results.sort(key=lambda item: item["combined_score"], reverse=True)
    return results

def find_relevant_sentences(question, results):
    q_terms = tokenize(question) - {
        "what", "how", "does", "is", "the", "a", "an", "and", "of", "for",
        "in", "on", "to", "it", "main", "advantage", "work", "explain"
    }
    candidates = []
    for result in results:
        # Split on sentence boundaries and line breaks; retain table-like short lines.
        pieces = re.split(r"(?<=[.!?])\s+|\n+", result["text"])
        for piece in pieces:
            sentence = re.sub(r"\s+", " ", piece).strip(" \t-•")
            if len(sentence) < 12:
                continue
            terms = tokenize(sentence)
            overlap = len(q_terms & terms) / max(1, len(q_terms))
            semantic_bonus = result["semantic_score"]
            # Prefer sentences that mention the terms asked about.
            score = (overlap * 0.75) + (semantic_bonus * 0.25)
            candidates.append({
                "text": sentence,
                "page": result["page"],
                "score": score,
                "overlap": overlap,
            })
    # Remove duplicate sentences while preserving the highest-ranked version.
    unique = {}
    for item in candidates:
        key = item["text"].lower()
        if key not in unique or item["score"] > unique[key]["score"]:
            unique[key] = item
    ranked = sorted(unique.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:3]

uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_pdf is not None:
    try:
        with st.spinner("Reading the PDF..."):
            page_texts, page_count = extract_pdf_text(uploaded_pdf)

        if not page_texts:
            st.error(
                "No selectable text was found. This PDF may be scanned. "
                "OCR is not included in this version."
            )
            st.stop()

        passages = split_into_passages(page_texts)
        if not passages:
            st.error("No text passages could be created from this PDF.")
            st.stop()

        with st.spinner("Creating the document search index..."):
            embedding_model = load_embedding_model()
            index = build_index(passages, embedding_model)

        st.success(
            f"PDF ready: {page_count} pages | {len(passages)} passages indexed"
        )

        # Clear prior chat when a different PDF is uploaded.
        if st.session_state.get("active_pdf_name") != uploaded_pdf.name:
            st.session_state.messages = []
            st.session_state.active_pdf_name = uploaded_pdf.name

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("Show retrieved source text"):
                        for n, source in enumerate(message["sources"], start=1):
                            st.markdown(f"**Source {n} · page {source['page']}**")
                            st.write(source["text"])

        question = st.chat_input("Ask a question about this PDF...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching for the most relevant passages..."):
                    results = retrieve_passages(
                        question, passages, embedding_model, index, top_k=5
                    )
                    relevant = find_relevant_sentences(question, results)

                if not relevant or max(item["score"] for item in relevant) < 0.18:
                    answer = (
                        "I could not find a clear answer in the uploaded document. "
                        "Try rephrasing the question or check the source text below."
                    )
                else:
                    answer = "Here is the most relevant information I found in your PDF:\n\n"
                    answer += "\n\n".join(
                        f"- {item['text']} *(Page {item['page']})*"
                        for item in relevant
                    )
                    answer += (
                        "\n\n*This answer is based on retrieved document text; "
                        "please verify important medical information with a qualified clinician.*"
                    )

                st.markdown(answer)
                with st.expander("Show retrieved source text"):
                    for n, source in enumerate(results, start=1):
                        st.markdown(
                            f"**Source {n} · page {source['page']} · "
                            f"similarity {source['semantic_score']:.3f}**"
                        )
                        st.text(source["text"][:2200])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": results,
            })

    except Exception:
        st.error(
            "The PDF could not be processed. Please try a text-based PDF. "
            "If the problem continues, check the Streamlit app logs."
        )
else:
    st.info("Upload a sample or public PDF to get started.")
