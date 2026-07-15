"""
DocuChat - A RAG-powered document Q&A assistant
Upload a PDF, build embeddings, and chat with your document.
"""

import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuChat — Chat with your Documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PERSIST_DIR = "./chroma_db"

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }

    .app-header {
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        color: white;
        margin: 0;
        font-size: 1.9rem;
    }
    .app-header p {
        color: #e0e7ff;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-ready {
        background-color: #064e3b;
        color: #6ee7b7;
    }
    .status-empty {
        background-color: #451a03;
        color: #fcd34d;
    }

    .doc-card {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        background-color: #1e2130;
        border: 1px solid #2d3348;
        margin-top: 0.6rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    section[data-testid="stSidebar"] {
        background-color: #131722;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "vector_store": None,
    "messages": [],
    "doc_name": None,
    "num_pages": 0,
    "num_chunks": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Cached / helper functions
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def process_pdf(uploaded_file):
    """Save the uploaded PDF, split it, embed it, and build a fresh Chroma store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)

        embedding_model = get_embedding_model()

        # Wipe any previous collection so old and new books never mix
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIR,
        )
        return vector_store, len(docs), len(chunks)
    finally:
        os.remove(tmp_path)


def build_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,   # out of 10, select 4
            "lambda_mult": 0.5,     # diversity
        },
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0,
    )

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a helpful AI assistant. You will be provided with a document and a question.
        Your task is to answer the question based on the document provided.
        Use only the provided context to answer the question. If the answer is not contained within the text below,
        say "I couldn't find the answer in the provided document\"""",
            ),
            (
                "human",
                """
        context : {context},
        question : {question}""",
            ),
        ]
    )

    return retriever, llm, template


def answer_question(query, retriever, llm, template):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = template.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)

    if isinstance(response.content, str):
        answer_text = response.content
    else:
        answer_text = "".join(
            block.get("text", "") for block in response.content if isinstance(block, dict)
        )
    return answer_text, docs


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>📚 DocuChat</h1>
        <p>Upload a PDF and ask questions about it — answers are grounded in your document.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — upload & status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("1. Upload your document")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    process_clicked = st.button(
        "🚀 Process document",
        use_container_width=True,
        disabled=uploaded_file is None,
        type="primary",
    )

    if process_clicked and uploaded_file is not None:
        with st.spinner("Reading, chunking, and embedding your document..."):
            vector_store, num_pages, num_chunks = process_pdf(uploaded_file)
        st.session_state.vector_store = vector_store
        st.session_state.doc_name = uploaded_file.name
        st.session_state.num_pages = num_pages
        st.session_state.num_chunks = num_chunks
        st.session_state.messages = []
        st.success("Document processed! Start chatting below.")

    st.markdown("---")
    st.subheader("Status")
    if st.session_state.vector_store is not None:
        st.markdown('<span class="status-pill status-ready">Ready</span>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="doc-card">
                <b>📄 {st.session_state.doc_name}</b><br>
                <span style="color:#9ca3af;font-size:0.85rem;">
                    {st.session_state.num_pages} pages · {st.session_state.num_chunks} chunks
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="status-pill status-empty">No document loaded</span>', unsafe_allow_html=True)
        st.caption("Upload and process a PDF to begin.")

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------
if st.session_state.vector_store is None:
    st.info("👈 Upload a PDF in the sidebar and click **Process document** to get started.")
else:
    # Replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources used"):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} (page {page}):**")
                        st.caption(src.page_content[:400] + "...")

    query = st.chat_input("Ask a question about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    retriever, llm, template = build_chain(st.session_state.vector_store)
                    answer_text, docs = answer_question(query, retriever, llm, template)
                except Exception as e:
                    answer_text = f"⚠️ Something went wrong: {e}"
                    docs = []

            st.markdown(answer_text)
            if docs:
                with st.expander("📎 Sources used"):
                    for i, src in enumerate(docs, 1):
                        page = src.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} (page {page}):**")
                        st.caption(src.page_content[:400] + "...")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "sources": docs}
        )