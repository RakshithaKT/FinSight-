"""
app.py — FinSight Streamlit UI

Run with:
    streamlit run app.py
"""

import streamlit as st
import time
from pathlib import Path
from main import run_analysis
from core.ingestor import load_and_split_pdfs, build_vectorstore, VECTORSTORE_DIR

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark financial terminal aesthetic ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e17;
    color: #c9d1d9;
}

/* Main header */
.fin-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-left: 4px solid #00d395;
    padding: 2rem 2.5rem;
    border-radius: 8px;
    margin-bottom: 2rem;
}

.fin-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #00d395;
    margin: 0;
    letter-spacing: -0.5px;
}

.fin-header p {
    color: #8b949e;
    margin: 0.4rem 0 0 0;
    font-size: 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* Cards */
.fin-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Report output */
.report-section {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 2rem;
    font-family: 'IBM Plex Sans', sans-serif;
    line-height: 1.8;
    white-space: pre-wrap;
}

/* Status badges */
.badge-green {
    background: #1a3a2a;
    color: #00d395;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    border: 1px solid #00d39544;
}

.badge-yellow {
    background: #332b00;
    color: #e3b341;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    border: 1px solid #e3b34144;
}

/* Metric boxes */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-box {
    flex: 1;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    text-align: center;
}

.metric-label {
    font-size: 0.75rem;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #00d395;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
}

/* Streamlit overrides */
.stButton > button {
    background: #00d395 !important;
    color: #0a0e17 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #00b37d !important;
    transform: translateY(-1px) !important;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stFileUploader {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 8px;
    padding: 1rem;
}

div[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}

.stSpinner > div {
    border-top-color: #00d395 !important;
}

hr {
    border-color: #21262d !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fin-header">
    <h1>📊 FinSight Agent</h1>
    <p>Autonomous Financial Report Analyst · LangGraph · RAG · Groq LLaMA 3</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Status")

    # Check vectorstore
    vs_exists = (VECTORSTORE_DIR / "index.faiss").exists()
    if vs_exists:
        st.markdown('<span class="badge-green">● FAISS Index Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-yellow">○ No FAISS Index</span>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📁 Ingest PDFs")
    st.markdown(
        "<small style='color:#8b949e'>Upload SEC 10-K filings or annual reports. "
        "Place them in the <code>/data</code> folder first.</small>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        for f in uploaded_files:
            save_path = data_dir / f.name
            save_path.write_bytes(f.read())
        st.success(f"Saved {len(uploaded_files)} PDF(s) to /data")

    if st.button("🔄 Build FAISS Index"):
        with st.spinner("Ingesting PDFs and building vector store..."):
            try:
                docs = load_and_split_pdfs()
                build_vectorstore(docs)
                st.success(f"✅ Indexed {len(docs)} chunks")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    st.markdown("### 🔑 Quick Setup")
    st.markdown("""
<small style='color:#8b949e'>
1. Add <code>GROQ_API_KEY</code> to <code>.env</code><br>
   → Free at <a href='https://console.groq.com' style='color:#00d395'>console.groq.com</a><br><br>
2. Add <code>TAVILY_API_KEY</code> to <code>.env</code><br>
   → Free at <a href='https://tavily.com' style='color:#00d395'>tavily.com</a><br><br>
3. Pull embedding model:<br>
   <code>ollama pull nomic-embed-text</code>
</small>
""", unsafe_allow_html=True)


# ── Main Area ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="metric-box">
        <div class="metric-label">Architecture</div>
        <div class="metric-value" style="font-size:1rem">4-Agent</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="metric-box">
        <div class="metric-label">Time Saved</div>
        <div class="metric-value">~75%</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="metric-box">
        <div class="metric-label">LLM</div>
        <div class="metric-value" style="font-size:1rem">LLaMA 3</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Query Input ────────────────────────────────────────────────────────────────
st.markdown("### 🔍 Analyze a Company")

example_queries = [
    "Apple Inc annual report 2023 revenue and profitability",
    "Infosys FY2024 financial performance and outlook",
    "Tesla 2023 10-K earnings and risk factors",
    "TCS revenue growth and margins 2024",
]

query_col, btn_col = st.columns([4, 1])
with query_col:
    query = st.text_input(
        "Enter company name or financial question",
        placeholder="e.g. Apple Inc Q3 2024 earnings analysis",
        label_visibility="collapsed",
    )

with btn_col:
    run_btn = st.button("▶ Run Analysis", use_container_width=True)

# Quick example buttons
st.markdown("<small style='color:#8b949e'>Quick examples:</small>", unsafe_allow_html=True)
ex_cols = st.columns(len(example_queries))
for i, (col, example) in enumerate(zip(ex_cols, example_queries)):
    with col:
        if st.button(example[:30] + "…", key=f"ex_{i}", use_container_width=True):
            query = example
            run_btn = True

# ── Run ────────────────────────────────────────────────────────────────────────
if run_btn and query:
    if not vs_exists:
        st.warning(
            "⚠️ No FAISS index found. Upload PDFs and click **Build FAISS Index** "
            "in the sidebar first. The agent will still run but without document context."
        )

    st.markdown("---")

    # Progress display
    progress_placeholder = st.empty()
    progress_placeholder.markdown("""
<div class="fin-card">
    <span class="badge-yellow">● RUNNING</span>
    <span style="margin-left:0.8rem;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:#8b949e">
        Initializing agents...
    </span>
</div>""", unsafe_allow_html=True)

    start_time = time.time()

    with st.spinner(""):
        try:
            # Agent status updates (best effort — Streamlit can't easily do true streaming)
            progress_placeholder.markdown("""
<div class="fin-card">
    <span class="badge-yellow">● RUNNING</span>
    <span style="margin-left:0.8rem;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:#8b949e">
        [1/3] RAG Agent — searching vector store...
    </span>
</div>""", unsafe_allow_html=True)

            result = run_analysis(query)
            elapsed = time.time() - start_time

        except Exception as e:
            progress_placeholder.error(f"Pipeline error: {e}")
            st.stop()

    # ── Results ────────────────────────────────────────────────────────────────
    if result.get("report"):
        progress_placeholder.markdown(f"""
<div class="fin-card">
    <span class="badge-green">● COMPLETE</span>
    <span style="margin-left:0.8rem;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:#8b949e">
        Report generated in {elapsed:.1f}s
    </span>
</div>""", unsafe_allow_html=True)

        # Tabs for report, sources, raw state
        tab1, tab2, tab3 = st.tabs(["📄 Analyst Report", "📚 Sources", "🔧 Debug"])

        with tab1:
            st.markdown("### Analyst Report")
            st.markdown(
                f'<div class="report-section">{result["report"]}</div>',
                unsafe_allow_html=True,
            )
            # Download button
            st.download_button(
                "⬇️ Download Report",
                data=result["report"],
                file_name=f"finsight_{query[:30].replace(' ','_')}.txt",
                mime="text/plain",
            )

        with tab2:
            doc_count = len(result.get("documents", []))
            web_count = len(result.get("web_results", []))
            st.markdown(f"**{doc_count} document chunks** from FAISS · **{web_count} web results** from Tavily")

            if result.get("documents"):
                with st.expander(f"📄 Document Sources ({doc_count} chunks)"):
                    for i, doc in enumerate(result["documents"], 1):
                        st.markdown(f"**Chunk {i}**")
                        st.markdown(
                            f"<div style='background:#161b22;border:1px solid #21262d;"
                            f"border-radius:6px;padding:0.8rem;font-size:0.82rem;"
                            f"font-family:IBM Plex Mono,monospace;color:#8b949e;"
                            f"white-space:pre-wrap'>{doc}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("")

            if result.get("web_results"):
                with st.expander(f"🌐 Web Results ({web_count} results)"):
                    for i, res in enumerate(result["web_results"], 1):
                        st.markdown(f"**Result {i}**")
                        st.markdown(
                            f"<div style='background:#161b22;border:1px solid #21262d;"
                            f"border-radius:6px;padding:0.8rem;font-size:0.82rem;"
                            f"font-family:IBM Plex Mono,monospace;color:#8b949e;"
                            f"white-space:pre-wrap'>{res}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("")

        with tab3:
            st.markdown("**Agent State (final)**")
            debug_state = {k: v for k, v in result.items() if k != "messages"}
            debug_state["documents"] = f"[{len(result.get('documents', []))} chunks]"
            debug_state["web_results"] = f"[{len(result.get('web_results', []))} results]"
            debug_state["report"] = f"[{len(result.get('report', '') or '')} chars]"
            st.json(debug_state)

    else:
        progress_placeholder.error(
            f"⚠️ Analysis failed: {result.get('error', 'Unknown error')}"
        )

elif run_btn and not query:
    st.warning("Please enter a company name or question first.")