import os
import re
import io
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from groq import Groq

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 200
RETRIEVAL_K     = 20
FINAL_K         = 10
USE_REWRITE     = True
USE_RERANK      = True
CHAT_MODEL      = "llama-3.3-70b-versatile"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_DIR          = "chroma_db_upload"
COLLECTION_NAME = "uploaded_docs"

SYSTEM_PROMPT = """You are a helpful multilingual assistant that answers questions based on the uploaded documents.
Detect the language of the user's question and ALWAYS reply in that same language.
Use ONLY the context below. If the context does not contain enough information, say so clearly in the user's language.
Be accurate, relevant, and complete.

Context:
{context}

Answer the user's question based on this context. Always respond in the same language as the question."""

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file.")
        st.stop()
    return Groq(api_key=api_key)

def load_file(uploaded_file) -> list:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(tmp_path)
        else:
            st.error(f"Unsupported file type: {suffix}")
            return []
        docs = loader.load()
    finally:
        os.unlink(tmp_path)
    return docs

def build_vectorstore(documents: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        return None, 0
    embeddings = get_embeddings()
    if os.path.exists(DB_DIR):
        try:
            Chroma(
                persist_directory=DB_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            ).delete_collection()
        except Exception:
            pass
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore, len(chunks)

def groq_chat(messages: list) -> str:
    client = get_groq_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""

def rewrite_query(question: str, history: list) -> str:
    history_text = "None"
    if history:
        parts = [f"User: {m['user']}\nAssistant: {m['assistant']}" for m in history]
        history_text = "\n".join(parts)
    prompt = (
        f"You are about to search a document to answer a user question.\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Current question: {question}\n\n"
        f"Reply with ONLY a short, specific search query (one line). No other text."
    )
    result = groq_chat([{"role": "user", "content": prompt}])
    return (result or question).strip().split("\n")[0].strip()

def rerank_chunks(question: str, chunks: list) -> list:
    if len(chunks) <= 1:
        return chunks
    prompt = (
        f"Question: {question}\n\n"
        "Rank these chunks by relevance (most relevant first). "
        "Reply with only the chunk numbers comma-separated (e.g. 3,1,2).\n\nChunks:\n"
    )
    for i, doc in enumerate(chunks):
        prompt += f"\n--- Chunk {i+1} ---\n{doc.page_content[:800]}\n"
    prompt += "\nReply with only comma-separated numbers."
    out = groq_chat([{"role": "user", "content": prompt}])
    numbers = re.findall(r"\d+", (out or "").strip())
    order = [int(n) - 1 for n in numbers if 1 <= int(n) <= len(chunks)]
    seen, ordered = set(), []
    for i in order:
        if i not in seen:
            seen.add(i)
            ordered.append(chunks[i])
    for i, c in enumerate(chunks):
        if i not in seen:
            ordered.append(c)
    return ordered[:FINAL_K] if ordered else chunks[:FINAL_K]

def merge_chunks(a, b):
    seen, out = set(), []
    for doc in a + b:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            out.append(doc)
    return out

def fetch_context(vectorstore, question: str, history: list) -> list:
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    chunks = retriever.invoke(question)
    if USE_REWRITE:
        rewritten = rewrite_query(question, history)
        if rewritten and rewritten != question:
            chunks = merge_chunks(chunks, retriever.invoke(rewritten))
    if USE_RERANK and chunks:
        chunks = rerank_chunks(question, chunks)
    else:
        chunks = chunks[:FINAL_K]
    return chunks

def answer_question(vectorstore, question: str, history: list) -> str:
    docs = fetch_context(vectorstore, question, history)
    context = "\n\n".join(
        f"From {doc.metadata.get('source', 'doc')}:\n{doc.page_content}" for doc in docs
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    for m in history:
        messages.append({"role": "user",      "content": m["user"]})
        messages.append({"role": "assistant", "content": m["assistant"]})
    messages.append({"role": "user", "content": question})
    return groq_chat(messages)

# ── PDF Export ────────────────────────────────────────────────────────────────
def export_chat_to_pdf(chat_history: list, doc_name: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title", parent=styles["Normal"],
        fontSize=16, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=4, alignment=TA_CENTER,
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER, spaceAfter=16,
    )
    user_label = ParagraphStyle(
        "UserLabel", parent=styles["Normal"],
        fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a73e8"), spaceAfter=2,
    )
    user_text = ParagraphStyle(
        "UserText", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=10, leftIndent=8,
    )
    ai_label = ParagraphStyle(
        "AiLabel", parent=styles["Normal"],
        fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f9d58"), spaceAfter=2,
    )
    ai_text = ParagraphStyle(
        "AiText", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#333333"),
        spaceAfter=14, leftIndent=8,
    )

    story = []
    story.append(Paragraph("RAG Chat Export", title_style))
    story.append(Paragraph(
        f"Document: {doc_name} &nbsp;&nbsp;|&nbsp;&nbsp; {datetime.now().strftime('%d %B %Y, %H:%M')}",
        meta_style,
    ))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))
    story.append(Spacer(1, 0.4*cm))

    for i, msg in enumerate(chat_history):
        # sanitize for reportlab XML
        def safe(text):
            return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
        story.append(Paragraph("You", user_label))
        story.append(Paragraph(safe(msg["user"]), user_text))
        story.append(Paragraph("Assistant", ai_label))
        story.append(Paragraph(safe(msg["assistant"]), ai_text))
        if i < len(chat_history) - 1:
            story.append(HRFlowable(width="100%", thickness=0.3,
                                    color=colors.HexColor("#eeeeee"), spaceAfter=6))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("RAG Chat - Ask Questions on Your Documents")
st.caption(f"Powered by Groq ({CHAT_MODEL}) + Multilingual HuggingFace Embeddings")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Document")
    uploaded_files = st.file_uploader(
        "Upload PDF or Word files (auto-processed)",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
    )

    # Auto-process when new files are uploaded
    if uploaded_files:
        file_keys = {f.name + str(f.size) for f in uploaded_files}
        if file_keys != st.session_state.processed_files:
            with st.spinner("Processing documents..."):
                all_docs = []
                for f in uploaded_files:
                    docs = load_file(f)
                    all_docs.extend(docs)
                if all_docs:
                    vs, n_chunks = build_vectorstore(all_docs)
                    if vs:
                        st.session_state.vectorstore = vs
                        st.session_state.chat_history = []
                        st.session_state.doc_name = ", ".join(f.name for f in uploaded_files)
                        st.session_state.processed_files = file_keys
                        st.success(f"Indexed {n_chunks} chunks!")
                    else:
                        st.error("No text could be extracted from the uploaded files.")

    st.divider()

    # Export chat as PDF
    if st.session_state.chat_history:
        pdf_bytes = export_chat_to_pdf(
            st.session_state.chat_history,
            st.session_state.doc_name or "document",
        )
        st.download_button(
            label="Export Chat as PDF",
            data=pdf_bytes,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown("**Settings**")
    st.markdown(f"- Model: `{CHAT_MODEL}`")
    st.markdown("- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`")
    st.markdown("- Languages: 50+ supported")
    st.markdown(f"- Query Rewrite: `{USE_REWRITE}`")
    st.markdown(f"- Reranking: `{USE_RERANK}`")

# ── Main Chat ─────────────────────────────────────────────────────────────────
if st.session_state.vectorstore is None:
    st.info("Upload a PDF or Word document in the sidebar to get started. It will be processed automatically.")
else:
    st.success(f"Active document(s): **{st.session_state.doc_name}**")

    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(msg["user"])
        with st.chat_message("assistant"):
            st.write(msg["assistant"])

    if question := st.chat_input("Ask a question in any language..."):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = answer_question(
                    st.session_state.vectorstore,
                    question,
                    st.session_state.chat_history,
                )
            st.write(reply)
        st.session_state.chat_history.append({"user": question, "assistant": reply})
        st.rerun()