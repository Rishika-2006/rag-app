import streamlit as st
import os
import tempfile
from rag_engine import load_documents, chunk_documents, create_vector_store, build_qa_chain, ask_question, stream_answer, load_url

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background-color: #1a1d2e;
        border-right: 1px solid #2d2f45;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6c63ff, #48cae4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #888;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        margin: 0.6rem 0;
        font-size: 15px;
        line-height: 1.7;
        max-width: 85%;
    }
    .user-message {
        background: linear-gradient(135deg, #6c63ff22, #6c63ff44);
        border: 1px solid #6c63ff66;
        color: #e0e0e0 !important;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .assistant-message {
        background: linear-gradient(135deg, #1e2235, #252840);
        border: 1px solid #2d2f45;
        color: #e0e0e0 !important;
        border-bottom-left-radius: 4px;
    }

    .role-label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        text-transform: uppercase;
    }
    .user-label { color: #6c63ff; text-align: right; }
    .assistant-label { color: #48cae4; }

    .source-box {
        background-color: #1a1d2e;
        border-left: 3px solid #48cae4;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-size: 13px;
        margin-top: 0.4rem;
        color: #aaa !important;
    }

    .status-ready {
        background-color: #1a3a2a;
        color: #4caf50 !important;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        border: 1px solid #4caf5044;
    }
    .status-waiting {
        background-color: #3a1a1a;
        color: #ef5350 !important;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-size: 13px;
        border: 1px solid #ef535044;
    }

    .doc-pill {
        background-color: #252840;
        border: 1px solid #6c63ff44;
        color: #c0b8ff !important;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px;
        display: inline-block;
    }

    .url-pill {
        background-color: #1a2a35;
        border: 1px solid #48cae444;
        color: #90e0ef !important;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px;
        display: inline-block;
    }

    .welcome-box {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 1px solid #2d2f45;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: #888 !important;
        margin-top: 2rem;
    }
    .welcome-icon { font-size: 3rem; margin-bottom: 1rem; }
    .welcome-title { color: #e0e0e0 !important; font-size: 1.2rem; font-weight: 600; }

    .stat-card {
        background-color: #1a1d2e;
        border: 1px solid #2d2f45;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .stat-number { font-size: 1.5rem; font-weight: 700; color: #6c63ff !important; }
    .stat-label { font-size: 12px; color: #888 !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "url_list" not in st.session_state:
    st.session_state.url_list = []

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAG Assistant")
    st.markdown("---")

    # File upload
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple files at once!"
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            size_kb = round(f.size / 1024, 1)
            st.markdown(f'<div class="doc-pill">📄 {f.name} ({size_kb} KB)</div>', unsafe_allow_html=True)

    st.markdown("")

    # URL input
    st.markdown("### 🌐 Or Load from URL")
    url_input = st.text_input(
        "Paste a website URL",
        placeholder="https://example.com/article",
        help="The app will read the webpage content"
    )
    add_url = st.button("➕ Add URL", use_container_width=True)

    if add_url and url_input:
        if url_input not in st.session_state.url_list:
            st.session_state.url_list.append(url_input)
            st.success("✅ URL added!")
        else:
            st.warning("URL already added!")

    if st.session_state.url_list:
        st.markdown("**Added URLs:**")
        for url in st.session_state.url_list:
            st.markdown(f'<div class="url-pill">🌐 {url[:35]}...</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear URLs", use_container_width=True):
            st.session_state.url_list = []
            st.rerun()

    st.markdown("")

    # Process button
    if uploaded_files or st.session_state.url_list:
        if st.button("⚙️ Process All Sources", use_container_width=True, type="primary"):
            with st.spinner("Reading and indexing all sources..."):
                try:
                    all_chunks = []
                    doc_names = []

                    # Process uploaded files
                    for uploaded_file in uploaded_files:
                        suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        docs = load_documents(tmp_path)
                        chunks = chunk_documents(docs)
                        all_chunks.extend(chunks)
                        doc_names.append(f"📄 {uploaded_file.name}")
                        os.unlink(tmp_path)

                    # Process URLs
                    for url in st.session_state.url_list:
                        with st.spinner(f"Loading {url[:30]}..."):
                            docs = load_url(url)
                            chunks = chunk_documents(docs)
                            all_chunks.extend(chunks)
                            doc_names.append(f"🌐 {url[:35]}...")

                    if not all_chunks:
                        st.warning("No content found! Try a different file or URL.")
                    else:
                        vector_store = create_vector_store(all_chunks)
                        st.session_state.qa_chain = build_qa_chain(vector_store)
                        st.session_state.uploaded_docs = doc_names
                        st.session_state.total_chunks = len(all_chunks)
                        st.session_state.chat_history = []
                        st.success(f"✅ {len(doc_names)} source(s) indexed!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")

    # Status
    if st.session_state.qa_chain:
        st.markdown('<div class="status-ready">🟢 Ready to chat</div>', unsafe_allow_html=True)
        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.uploaded_docs)}</div>
                <div class="stat-label">Sources</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{st.session_state.total_chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("**Loaded sources:**")
        for doc in st.session_state.uploaded_docs:
            st.markdown(f'<div class="doc-pill">{doc}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-waiting">🔴 No sources loaded</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Reset Everything", use_container_width=True):
        st.session_state.qa_chain = None
        st.session_state.chat_history = []
        st.session_state.uploaded_docs = []
        st.session_state.total_chunks = 0
        st.session_state.url_list = []
        st.rerun()

    st.markdown("")
    st.caption("Built with LangChain · FAISS · Groq · HuggingFace")

# ── Main area ─────────────────────────────────────────────────
st.markdown('<p class="app-title">🧠 RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Chat with your documents and websites using AI</p>', unsafe_allow_html=True)

if not st.session_state.qa_chain:
    st.markdown("""
    <div class="welcome-box">
        <div class="welcome-icon">🧠</div>
        <div class="welcome-title">No sources loaded yet</div>
        <p>Upload PDF or TXT files, or paste a website URL from the sidebar,<br>
        then click <b>Process All Sources</b> and start asking questions!</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown('<div class="role-label user-label">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message user-message">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="role-label assistant-label">Assistant</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📄 View sources", expanded=False):
                    for i, doc in enumerate(msg["sources"]):
                        page = doc.metadata.get("page", "N/A")
                        source = doc.metadata.get("source", "Unknown")
                        snippet = doc.page_content[:200].strip()
                        st.markdown(f'<div class="source-box"><b>Source {i+1}</b> — {os.path.basename(source)} | Page {page}<br><br>{snippet}...</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Question input
    question = st.chat_input("Ask anything about your documents or URLs...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        try:
            sources = st.session_state.qa_chain["retriever"].invoke(question)

            st.markdown('<div class="role-label assistant-label">Assistant</div>', unsafe_allow_html=True)
            with st.chat_message("assistant"):
                streamed_answer = st.write_stream(
                    stream_answer(st.session_state.qa_chain, question)
                )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": streamed_answer,
                "sources": sources
            })

        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Sorry, something went wrong: {str(e)}",
                "sources": []
            })

        st.rerun()