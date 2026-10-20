import streamlit as st
import numpy as np
import torch
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="AI Healthcare Document Assistant",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 AI Healthcare Document Assistant")
st.caption("Upload a PDF and ask questions about its contents.")

st.warning(
    "For learning and demonstration only. Do not upload real patient-identifiable "
    "records or confidential hospital documents. This app does not provide a medical diagnosis."
)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_answer_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_no}]\n{text}")
    return "\n\n".join(pages), len(reader.pages)

def make_chunks(text, chunk_size=900, overlap=150):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks

def retrieve_chunks(question, chunks, embedding_model, index, top_k=3):
    q_vector = embedding_model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vector)
    k = min(top_k, len(chunks))
    scores, ids = index.search(q_vector, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            results.append((float(score), chunks[int(idx)]))
    return results

def generate_answer(question, retrieved_chunks, tokenizer, model):
    context = "\n\n".join(chunk for _, chunk in retrieved_chunks)
    prompt = (
        "Answer the question using only the document context below. "
        "If the answer is not stated in the context, say: "
        "'I could not find this information in the uploaded document.' "
        "Do not guess. For medical reports, explain what the document says "
        "but do not diagnose or recommend treatment.\n\n"
        f"Document context:\n{context[:4500]}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=140,
            num_beams=2,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_pdf is not None:
    try:
        with st.spinner("Reading the PDF..."):
            pdf_text, page_count = extract_pdf_text(uploaded_pdf)

        if not pdf_text.strip():
            st.error(
                "No selectable text was found. This may be a scanned PDF; "
                "OCR support is not included in this first version."
            )
            st.stop()

        chunks = make_chunks(pdf_text)
        if not chunks:
            st.error("No text chunks could be created from this PDF.")
            st.stop()

        with st.spinner("Loading embedding model and indexing the document..."):
            embedding_model = load_embedding_model()
            vectors = embedding_model.encode(
                chunks,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)

        st.success(
            f"PDF ready: {page_count} pages | {len(chunks)} text chunks indexed"
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if st.session_state.get("active_pdf_name") != uploaded_pdf.name:
            st.session_state.messages = []
            st.session_state.active_pdf_name = uploaded_pdf.name

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about this PDF...")
        if question:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the document and preparing an answer..."):
                    results = retrieve_chunks(
                        question, chunks, embedding_model, index, top_k=3
                    )
                    tokenizer, answer_model = load_answer_model()
                    answer = generate_answer(
                        question, results, tokenizer, answer_model
                    )
                st.markdown(answer or "I could not generate an answer from this document.")
                with st.expander("Show retrieved source text"):
                    for rank, (score, chunk) in enumerate(results, start=1):
                        st.markdown(f"**Source {rank} · similarity {score:.3f}**")
                        st.text(chunk[:1800])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer or "I could not generate an answer from this document.",
                }
            )

    except Exception as exc:
        st.error("The PDF could not be processed. Please check the file and try again.")
        st.exception(exc)
else:
    st.info("Start by uploading a sample or public PDF. You can then ask questions about its contents.")
