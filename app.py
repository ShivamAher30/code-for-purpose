"""
app.py — Talk-to-Data AI Assistant
Main Streamlit application with all 20 hackathon features integrated.
"""

import os
import json
import hashlib
import pandas as pd
import streamlit as st
import torch
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────
# PAGE CONFIG & IMPORTS
# ──────────────────────────────────────────────

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

st.set_page_config(
    layout="wide",
    page_title="Talk-to-Data AI",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

from data_upload.input_sources_utils import image_util, pdf_util, website_util, audio_util
from data_upload.input_sources_utils import excel_util
from utils import (
    load_clip_model, load_text_embedding_model, load_whisper_model,
    load_text_index, load_audio_index, load_image_index,
    auto_detect_schema, detect_sensitive_columns, mask_sensitive_data,
)
from vectordb import search_text_index, search_image_index
from llm_engine import (
    call_llm, route_intent, nl_to_pandas, nl_to_pandas_with_retry,
    generate_explanation, generate_insights, generate_answer_from_pandas,
    check_query_clarity, narrate_comparison, narrate_root_cause,
    narrate_breakdown, narrate_summary, narrate_anomalies,
    detect_metric_and_dimensions, analyze_vision_chart,
)
from visualize import (
    execute_pandas_code_safely, generate_auto_chart, get_auto_explanation,
    generate_comparison_chart, generate_driver_chart, generate_anomaly_chart,
)
from analysis_engine import (
    root_cause_analysis, compare_periods, compare_segments,
    breakdown_metric, detect_anomalies, generate_data_summary, rank_insights,
)
from export_engine import export_to_pdf, export_dataframe_to_csv

# ──────────────────────────────────────────────
# PREMIUM CSS STYLING (Feature 13)
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #E94560 50%, #0ABAB5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    
    .sub-header {
        color: #888;
        font-size: 0.95rem;
        margin-top: -8px;
        margin-bottom: 20px;
        font-weight: 300;
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 16px !important;
        border: 1px solid rgba(108, 99, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(108, 99, 255, 0.15);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(135deg, #6C63FF, #E94560);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.4rem;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(108, 99, 255, 0.3);
        font-weight: 500;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        border-color: #6C63FF;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.2);
        transform: translateY(-1px);
    }
    
    /* Schema cards */
    .schema-card {
        background: linear-gradient(135deg, rgba(108,99,255,0.08), rgba(233,69,96,0.05));
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    
    .badge-numeric { background: rgba(108,99,255,0.15); color: #6C63FF; }
    .badge-categorical { background: rgba(233,69,96,0.15); color: #E94560; }
    .badge-datetime { background: rgba(10,186,181,0.15); color: #0ABAB5; }
    .badge-sensitive { background: rgba(255,107,53,0.15); color: #FF6B35; }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        border-radius: 10px;
        font-weight: 500;
    }
    
    /* Intent badge */
    .intent-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: linear-gradient(135deg, rgba(108,99,255,0.2), rgba(233,69,96,0.2));
        color: #6C63FF;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }
    
    /* Trust layer */
    .trust-expander {
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 12px;
    }
    
    /* Export buttons container */
    .export-container {
        display: flex;
        gap: 8px;
        margin-top: 10px;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(108,99,255,0.1), rgba(233,69,96,0.05));
        border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #E94560);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #888;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(108, 99, 255, 0.3);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_clip_model()
text_embedding_model = load_text_embedding_model()
whisper_model = load_whisper_model()

# Handle missing directories
_db_dirs = ["./vectorstore", "./images"]
for d in _db_dirs:
    Path(d).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "df" not in st.session_state:
    st.session_state["df"] = None

if "schema" not in st.session_state:
    st.session_state["schema"] = None

if "sensitive_cols" not in st.session_state:
    st.session_state["sensitive_cols"] = []

if "mask_sensitive" not in st.session_state:
    st.session_state["mask_sensitive"] = False

if "query_cache" not in st.session_state:
    st.session_state["query_cache"] = {}

if "pending_clarification" not in st.session_state:
    st.session_state["pending_clarification"] = None


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🧠 Talk-to-Data AI")
    st.caption("Upload data • Ask questions • Get insights")

    st.divider()

    # ── Section 1: Data Upload ──
    st.header("📁 Data Source")

    upload_tab_csv, upload_tab_excel = st.tabs(["CSV", "Excel"])

    with upload_tab_csv:
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
        if uploaded_csv is not None:
            try:
                st.session_state["df"] = pd.read_csv(uploaded_csv)
                st.session_state["schema"] = auto_detect_schema(st.session_state["df"])
                st.session_state["sensitive_cols"] = detect_sensitive_columns(st.session_state["df"])
                st.success(f"✅ {len(st.session_state['df'])} rows loaded")
            except Exception as e:
                st.error(f"Error: {e}")

    with upload_tab_excel:
        uploaded_excel = st.file_uploader("Upload Excel", type=["xlsx", "xls"], key="excel_uploader")
        if uploaded_excel is not None:
            try:
                xls = pd.ExcelFile(uploaded_excel)
                sheet_names = xls.sheet_names
                if len(sheet_names) > 1:
                    selected_sheet = st.selectbox("Select sheet", sheet_names)
                else:
                    selected_sheet = sheet_names[0]
                st.session_state["df"] = pd.read_excel(uploaded_excel, sheet_name=selected_sheet)
                st.session_state["schema"] = auto_detect_schema(st.session_state["df"])
                st.session_state["sensitive_cols"] = detect_sensitive_columns(st.session_state["df"])
                st.success(f"✅ {len(st.session_state['df'])} rows from '{selected_sheet}'")
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Schema Display (Feature 3) ──
    if st.session_state["schema"]:
        schema = st.session_state["schema"]
        with st.expander("📊 Data Schema & Auto-Detection", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"<div class='stat-card'><div class='stat-value'>{schema['total_rows']}</div><div class='stat-label'>Rows</div></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='stat-card'><div class='stat-value'>{schema['total_columns']}</div><div class='stat-label'>Columns</div></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='stat-card'><div class='stat-value'>{len(schema['date_columns'])}</div><div class='stat-label'>Date Cols</div></div>", unsafe_allow_html=True)

            st.markdown("**Column Types:**")
            for col in schema.get("numeric_columns", []):
                st.markdown(f"<span class='metric-badge badge-numeric'>📐 {col}</span>", unsafe_allow_html=True)
            for col in schema.get("categorical_columns", []):
                st.markdown(f"<span class='metric-badge badge-categorical'>🏷️ {col}</span>", unsafe_allow_html=True)
            for col in schema.get("date_columns", []):
                st.markdown(f"<span class='metric-badge badge-datetime'>📅 {col}</span>", unsafe_allow_html=True)

    # ── Privacy Controls (Feature 7) ──
    if st.session_state["sensitive_cols"]:
        with st.expander("🔒 Privacy & Security", expanded=False):
            st.warning(f"⚠️ {len(st.session_state['sensitive_cols'])} potentially sensitive columns detected:")
            for col_name, reason in st.session_state["sensitive_cols"]:
                st.markdown(f"<span class='metric-badge badge-sensitive'>🔐 {col_name}</span>", unsafe_allow_html=True)
                st.caption(reason)
            st.session_state["mask_sensitive"] = st.toggle("Mask Sensitive Data", value=st.session_state["mask_sensitive"], help="When enabled, PII columns are masked before sending to the AI.")

    st.divider()

    # ── Section 2: Quick Actions ──
    st.header("⚡ Quick Actions")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📋 Summary", use_container_width=True, disabled=st.session_state["df"] is None):
            st.session_state["_trigger_summary"] = True
    with col_b:
        if st.button("🔍 Anomalies", use_container_width=True, disabled=st.session_state["df"] is None):
            st.session_state["_trigger_anomaly"] = True

    col_c, col_d = st.columns(2)
    with col_c:
        if st.button("💡 Insights", use_container_width=True, disabled=st.session_state["df"] is None):
            st.session_state["_trigger_insights"] = True
    with col_d:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["query_cache"] = {}
            st.session_state["pending_clarification"] = None
            st.rerun()

    st.divider()

    # ── Section 3: Unstructured Data (RAG) ──
    st.header("📄 Unstructured Data")
    upload_choice = st.selectbox(
        options=["Upload PDF", "Website Link", "Upload Image", "Add Image from URL", "Audio Recording"],
        label="Upload Type"
    )
    if upload_choice == "Upload Image":
        image_util.upload_image(clip_model, preprocess)
    elif upload_choice == "Add Image from URL":
        image_util.image_from_url(clip_model, preprocess)
    elif upload_choice == "Upload PDF":
        pdf_util.upload_pdf(clip_model, preprocess, text_embedding_model)
    elif upload_choice == "Website Link":
        website_util.data_from_website(clip_model, preprocess, text_embedding_model)
    elif upload_choice == "Audio Recording":
        audio_util.upload_audio(whisper_model, text_embedding_model)

    st.divider()

    # ── Section 4: Routing Mode ──
    st.header("⚙️ Settings")
    routing_mode = st.radio(
        "Routing Mode",
        ["Auto-Detect", "Structured (CSV)", "Unstructured (RAG)"],
        index=0,
        help="Force AI to target specific data source."
    )

    st.divider()

    # ── Section 5: Semantic Dictionary (Feature 16) ──
    st.header("📖 Metric Dictionary")
    dict_path = "semantic_dict.json"
    metrics_dict = {}
    if os.path.exists(dict_path):
        try:
            with open(dict_path, "r") as f:
                metrics_dict = json.load(f)
        except:
            pass

    with st.expander("Define Business Metrics", expanded=False):
        st.caption("Ensure consistent metric calculations")
        if metrics_dict:
            for k, v in metrics_dict.items():
                st.markdown(f"**{k}**: {v}")
        else:
            st.write("No metrics defined.")

        st.divider()
        new_metric_name = st.text_input("Metric Name", placeholder="e.g., Gross Churn")
        new_metric_def = st.text_area("Definition", height=60, placeholder="e.g., Total churned / Total active * 100")
        if st.button("Add Metric", use_container_width=True):
            if new_metric_name and new_metric_def:
                metrics_dict[new_metric_name] = new_metric_def
                with open(dict_path, "w") as f:
                    json.dump(metrics_dict, f, indent=2)
                st.success(f"Added '{new_metric_name}'!")
                st.rerun()


# ──────────────────────────────────────────────
# MAIN CHAT INTERFACE
# ──────────────────────────────────────────────

st.markdown("<h1 class='main-header'>Talk-to-Data AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload your data and ask anything — get instant insights, charts, and explanations.</p>", unsafe_allow_html=True)

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "uploaded_images" in msg and msg["uploaded_images"]:
            for img_bytes in msg["uploaded_images"]:
                st.image(img_bytes, width=400)

        # Trust Layer (Feature 6)
        if "trust_layer" in msg and msg["trust_layer"]:
            with st.expander("🔍 Source Transparency & Explainability"):
                tl = msg["trust_layer"]
                if tl.get("intent"):
                    st.markdown(f"<span class='intent-badge'>🎯 {tl['intent'].upper()}</span>", unsafe_allow_html=True)
                if tl.get("pandas_code"):
                    st.subheader("Generated Code")
                    st.code(tl["pandas_code"], language="python")
                if tl.get("dataframe_preview") is not None:
                    st.subheader("Data Result")
                    try:
                        st.dataframe(tl["dataframe_preview"], use_container_width=True)
                    except Exception:
                        st.dataframe(tl["dataframe_preview"].astype(str), use_container_width=True)
                if tl.get("explanation"):
                    st.subheader("How It Was Computed")
                    st.write(tl["explanation"])
                if tl.get("sources"):
                    st.subheader("Source Chunks")
                    for idx, c in enumerate(tl["sources"]):
                        st.write(f"**Chunk {idx+1}:** {c}")
                        st.divider()
                if tl.get("analysis_data"):
                    st.subheader("Raw Analysis Data")
                    st.json(tl["analysis_data"])

        # Chart
        if "chart" in msg and msg["chart"] is not None:
            st.pyplot(msg["chart"])

        # Export Buttons (Feature 20)
        if msg["role"] == "assistant" and msg.get("content") and not msg["content"].startswith("Please upload"):
            exp_col1, exp_col2, _ = st.columns([1, 1, 4])
            with exp_col1:
                pdf_bytes = export_to_pdf(
                    title="Talk-to-Data Report",
                    query=msg.get("_query", ""),
                    response_text=msg["content"],
                    chart_fig=msg.get("chart"),
                    dataframe=msg.get("trust_layer", {}).get("dataframe_preview") if msg.get("trust_layer") else None,
                    trust_info=msg.get("trust_layer"),
                )
                if pdf_bytes:
                    st.download_button(
                        "📄 PDF", data=pdf_bytes,
                        file_name=f"insight_{datetime.now().strftime('%H%M%S')}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{hash(msg['content'][:50])}_{id(msg)}",
                    )
            with exp_col2:
                if msg.get("trust_layer", {}).get("dataframe_preview") is not None:
                    csv_bytes = export_dataframe_to_csv(msg["trust_layer"]["dataframe_preview"])
                    if csv_bytes:
                        st.download_button(
                            "📊 CSV", data=csv_bytes,
                            file_name=f"data_{datetime.now().strftime('%H%M%S')}.csv",
                            mime="text/csv",
                            key=f"csv_{hash(msg['content'][:50])}_{id(msg)}",
                        )


# ──────────────────────────────────────────────
# QUICK ACTION HANDLERS
# ──────────────────────────────────────────────

def handle_quick_action(action_type):
    """Process quick action buttons."""
    df = st.session_state["df"]
    if df is None:
        return

    with st.chat_message("assistant"):
        if action_type == "summary":
            with st.spinner("📋 Generating comprehensive summary..."):
                summary_data = generate_data_summary(df)
                narrative = narrate_summary("Give me a summary of this data", summary_data)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": narrative,
                    "trust_layer": {"intent": "summary", "analysis_data": summary_data},
                    "_query": "Quick Action: Auto Summary",
                })
                st.rerun()

        elif action_type == "anomaly":
            with st.spinner("🔍 Detecting anomalies..."):
                anomaly_data = detect_anomalies(df)
                narrative = narrate_anomalies("Detect anomalies in this data", anomaly_data)

                charts = []
                if anomaly_data.get("has_anomalies"):
                    for col, info in list(anomaly_data["anomalies"].items())[:2]:
                        fig = generate_anomaly_chart(df, col, info)
                        if fig:
                            charts.append(fig)

                msg_data = {
                    "role": "assistant",
                    "content": narrative,
                    "trust_layer": {"intent": "anomaly", "analysis_data": anomaly_data},
                    "chart": charts[0] if charts else None,
                    "_query": "Quick Action: Anomaly Detection",
                }
                st.session_state.messages.append(msg_data)
                st.rerun()

        elif action_type == "insights":
            with st.spinner("💡 Generating insights..."):
                insights = generate_insights(df)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": insights,
                    "trust_layer": {"intent": "structured"},
                    "_query": "Quick Action: Generic Insights",
                })
                st.rerun()


# Handle quick action triggers
if st.session_state.get("_trigger_summary"):
    del st.session_state["_trigger_summary"]
    handle_quick_action("summary")

if st.session_state.get("_trigger_anomaly"):
    del st.session_state["_trigger_anomaly"]
    handle_quick_action("anomaly")

if st.session_state.get("_trigger_insights"):
    del st.session_state["_trigger_insights"]
    handle_quick_action("insights")


# ──────────────────────────────────────────────
# CHAT INPUT HANDLER
# ──────────────────────────────────────────────

query_prompt = st.chat_input("Ask anything about your data...", accept_file="multiple", file_type=["png", "jpg", "jpeg"])

if query_prompt:
    if hasattr(query_prompt, "text"):
        query = query_prompt.text
        files = query_prompt.files
    else:
        query = query_prompt
        files = []

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        if files:
            for f in files:
                st.image(f, width=400)
                st.session_state.messages[-1].setdefault("uploaded_images", []).append(f.getvalue())

    # Check capabilities
    has_df = st.session_state["df"] is not None
    has_rag_text = os.path.exists("./vectorstore/text_index.index")
    has_rag_audio = os.path.exists("./vectorstore/audio_index.index")
    has_rag_image = os.path.exists("./vectorstore/image_index.index")
    has_rag = has_rag_text or has_rag_audio or has_rag_image

    if not has_df and not has_rag:
        with st.chat_message("assistant"):
            warning_msg = "👋 Welcome! Please upload some data (CSV, Excel, or documents) from the sidebar to get started."
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
        st.stop()

    # Determine intent
    if files:
        intent = "vision"
    elif routing_mode == "Structured (CSV)":
        intent = "structured"
    elif routing_mode == "Unstructured (RAG)":
        intent = "rag"
    else:
        intent = route_intent(query, has_df, has_rag)

    # Get working DataFrame (with masking if enabled)
    df = st.session_state["df"]
    if df is not None and st.session_state["mask_sensitive"] and st.session_state["sensitive_cols"]:
        df_for_llm = mask_sensitive_data(df, st.session_state["sensitive_cols"])
    else:
        df_for_llm = df

    # ── Cache Check (Feature 12) ──
    cache_key = hashlib.md5(f"{query}_{intent}".encode()).hexdigest()
    cached = st.session_state["query_cache"].get(cache_key)

    with st.chat_message("assistant"):
        if cached:
            st.markdown(f"<span class='intent-badge'>⚡ CACHED</span>", unsafe_allow_html=True)
            response_content = cached["content"]
            trust_layer = cached.get("trust_layer", {})
            chart_fig = cached.get("chart")
            st.markdown(response_content)
            if trust_layer:
                with st.expander("🔍 Source Transparency & Explainability"):
                    if trust_layer.get("pandas_code"):
                        st.code(trust_layer["pandas_code"], language="python")
                    if trust_layer.get("dataframe_preview") is not None:
                        st.dataframe(trust_layer["dataframe_preview"], use_container_width=True)
                    if trust_layer.get("explanation"):
                        st.write(trust_layer["explanation"])
            if chart_fig:
                st.pyplot(chart_fig)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_content,
                "trust_layer": trust_layer,
                "chart": chart_fig,
                "_query": query,
            })
        else:
            st.markdown(f"<span class='intent-badge'>🎯 {intent.upper().replace('_', ' ')}</span>", unsafe_allow_html=True)

            with st.spinner(f"Processing as **{intent.replace('_', ' ').title()}**..."):
                response_content = ""
                trust_layer = {"intent": intent}
                chart_fig = None
                retrieved_images = []

                # ════════════════════════════════════════
                # VISION INTENT
                # ════════════════════════════════════════
                if intent == "vision":
                    res = analyze_vision_chart(query, files[0].getvalue())
                    response_content = res
                    trust_layer["explanation"] = "Chart analyzed via Llama 3.2 Vision multimodal model."

                # ════════════════════════════════════════
                # STRUCTURED INTENT (Feature 2)
                # ════════════════════════════════════════
                elif intent == "structured":
                    if not has_df:
                        response_content = "You asked a structured query, but no CSV/Excel is uploaded. Please upload data first."
                    else:
                        # Clarification check (Feature 15)
                        is_clear, clarification_q = check_query_clarity(query, df_for_llm)
                        if not is_clear:
                            response_content = f"🤔 {clarification_q}"
                            trust_layer["explanation"] = "Query was ambiguous — requested clarification."
                        else:
                            history = st.session_state.messages[-10:]
                            # Error retry loop (Feature 2)
                            code, df_result, err = nl_to_pandas_with_retry(query, df_for_llm, history=history)
                            trust_layer["pandas_code"] = code

                            if err:
                                response_content = f"I encountered an issue processing that query. Error: {err}\n\nTry rephrasing your question or being more specific about which columns to use."
                                trust_layer["explanation"] = f"Code execution failed after retries: {err}"
                            else:
                                # Auto chart (always attempt — Feature 5)
                                chart_fig = generate_auto_chart(df_result, query_hint=query)

                                explanation = generate_explanation(query, None, df_result)
                                trust_layer["explanation"] = explanation

                                if isinstance(df_result, (pd.DataFrame, pd.Series)):
                                    trust_layer["dataframe_preview"] = df_result
                                    response_content = generate_answer_from_pandas(query, df_result)
                                else:
                                    response_content = generate_answer_from_pandas(query, df_result)

                # ════════════════════════════════════════
                # COMPARISON INTENT (Feature 9)
                # ════════════════════════════════════════
                elif intent == "comparison":
                    if not has_df:
                        response_content = "No data uploaded for comparison. Please upload a CSV or Excel file."
                    else:
                        metric_col, dim_cols, period_col = detect_metric_and_dimensions(query, df_for_llm)

                        if metric_col and period_col:
                            unique_periods = df_for_llm[period_col].dropna().unique()
                            if len(unique_periods) >= 2:
                                comp = compare_periods(df_for_llm, metric_col, period_col, unique_periods[-2], unique_periods[-1])
                                response_content = narrate_comparison(query, comp)
                                chart_fig = generate_comparison_chart(comp, metric_name=metric_col)
                                trust_layer["analysis_data"] = comp
                                trust_layer["explanation"] = f"Compared '{metric_col}' across '{period_col}': {unique_periods[-2]} vs {unique_periods[-1]}"
                            else:
                                response_content = f"Only one period found in '{period_col}'. Need at least two periods for comparison."
                        elif metric_col and dim_cols:
                            comp = compare_segments(df_for_llm, metric_col, dim_cols[0])
                            response_content = narrate_breakdown(query, {"metric": metric_col, "grand_total": comp["grand_total"], "breakdowns": {dim_cols[0]: {"segments": comp["segments"]}}})
                            trust_layer["analysis_data"] = comp

                            # Chart from comparison
                            seg_df = pd.DataFrame(comp["segments"])
                            if not seg_df.empty:
                                chart_fig = generate_auto_chart(seg_df.set_index(dim_cols[0])["total"], query_hint=query)
                            trust_layer["explanation"] = f"Compared '{metric_col}' across '{dim_cols[0]}'"
                        else:
                            # Fallback to structured
                            history = st.session_state.messages[-10:]
                            code, df_result, err = nl_to_pandas_with_retry(query, df_for_llm, history=history)
                            trust_layer["pandas_code"] = code
                            if err:
                                response_content = f"Could not execute comparison: {err}"
                            else:
                                chart_fig = generate_auto_chart(df_result, query_hint=query)
                                if isinstance(df_result, (pd.DataFrame, pd.Series)):
                                    trust_layer["dataframe_preview"] = df_result
                                response_content = generate_answer_from_pandas(query, df_result)

                # ════════════════════════════════════════
                # ROOT CAUSE INTENT (Feature 8)
                # ════════════════════════════════════════
                elif intent == "root_cause":
                    if not has_df:
                        response_content = "No data uploaded. Please upload a CSV or Excel file for root cause analysis."
                    else:
                        metric_col, dim_cols, period_col = detect_metric_and_dimensions(query, df_for_llm)

                        if metric_col:
                            rca = root_cause_analysis(df_for_llm, metric_col, period_col=period_col)
                            response_content = narrate_root_cause(query, rca)
                            trust_layer["analysis_data"] = rca

                            if "top_drivers" in rca:
                                chart_fig = generate_driver_chart(rca["top_drivers"], title=f"Root Cause: {metric_col}")
                            trust_layer["explanation"] = f"Root cause analysis on '{metric_col}' decomposed across {len(dim_cols)} dimensions"
                        else:
                            response_content = "I couldn't identify the metric to analyze. Please specify which numeric column you're asking about."

                # ════════════════════════════════════════
                # BREAKDOWN INTENT (Feature 10)
                # ════════════════════════════════════════
                elif intent == "breakdown":
                    if not has_df:
                        response_content = "No data uploaded. Please upload a CSV or Excel file."
                    else:
                        metric_col, dim_cols, period_col = detect_metric_and_dimensions(query, df_for_llm)

                        if metric_col:
                            bd = breakdown_metric(df_for_llm, metric_col, group_cols=dim_cols if dim_cols else None)
                            response_content = narrate_breakdown(query, bd)
                            trust_layer["analysis_data"] = bd

                            # Chart for first breakdown dimension
                            if bd.get("breakdowns"):
                                first_dim = list(bd["breakdowns"].keys())[0]
                                segs = bd["breakdowns"][first_dim]["segments"]
                                seg_df = pd.DataFrame(segs)
                                if not seg_df.empty and first_dim in seg_df.columns:
                                    chart_fig = generate_auto_chart(
                                        seg_df.set_index(first_dim)["total"],
                                        query_hint="breakdown share"
                                    )
                            trust_layer["explanation"] = f"Breakdown of '{metric_col}' across {len(bd.get('breakdowns', {}))} dimensions"
                        else:
                            response_content = "I couldn't identify the metric to break down. Please specify which numeric column you're asking about."

                # ════════════════════════════════════════
                # SUMMARY INTENT (Feature 11)
                # ════════════════════════════════════════
                elif intent == "summary":
                    if not has_df:
                        response_content = "No data uploaded. Please upload a CSV or Excel file."
                    else:
                        summary = generate_data_summary(df_for_llm)
                        response_content = narrate_summary(query, summary)
                        trust_layer["analysis_data"] = summary
                        trust_layer["explanation"] = "Comprehensive analysis covering all columns, distributions, and patterns."

                # ════════════════════════════════════════
                # ANOMALY INTENT (Feature 17)
                # ════════════════════════════════════════
                elif intent == "anomaly":
                    if not has_df:
                        response_content = "No data uploaded. Please upload a CSV or Excel file."
                    else:
                        anomalies = detect_anomalies(df_for_llm)
                        response_content = narrate_anomalies(query, anomalies)
                        trust_layer["analysis_data"] = anomalies

                        if anomalies.get("has_anomalies"):
                            first_col = list(anomalies["anomalies"].keys())[0]
                            chart_fig = generate_anomaly_chart(df_for_llm, first_col, anomalies["anomalies"][first_col])
                        trust_layer["explanation"] = f"IQR-based anomaly detection across {len(anomalies.get('anomalies', {}))} columns"

                # ════════════════════════════════════════
                # RAG INTENT
                # ════════════════════════════════════════
                else:
                    context_chunks = []

                    if has_rag_text:
                        try:
                            ti, td = load_text_index()
                            res_idxs = search_text_index(query, ti, text_embedding_model, k=3)
                            for idx in res_idxs[0]:
                                if 0 <= idx < len(td):
                                    context_chunks.append(td['content'].iloc[idx])
                        except Exception:
                            pass
                    if has_rag_audio:
                        try:
                            ai, ad = load_audio_index()
                            res_idxs = search_text_index(query, ai, text_embedding_model, k=2)
                            for idx in res_idxs[0]:
                                if 0 <= idx < len(ad):
                                    context_chunks.append(ad['content'].iloc[idx])
                        except Exception:
                            pass

                    visual_keywords = ["image", "picture", "photo", "show me", "visual", "graph", "look like"]
                    if has_rag_image and any(word in query.lower() for word in visual_keywords):
                        try:
                            ii, id_df = load_image_index()
                            res_idxs = search_image_index(query, ii, clip_model, k=1)
                            for idx in res_idxs[0]:
                                if 0 <= idx < len(id_df):
                                    img_path = id_df['path'].iloc[idx]
                                    retrieved_images.append(img_path)
                                    context_chunks.append(f"Image found matching query at path: {img_path}")
                        except Exception:
                            pass

                    if not context_chunks and not retrieved_images:
                        response_content = "I couldn't find relevant information in the uploaded documents. Try uploading more data or rephrasing your question."
                    else:
                        combined_context = " ".join(context_chunks)[:1500]
                        prompt = f"Answer the question based on the context below.\n\nContext: {combined_context}\n\nQuestion: {query}\n\nAnswer:"

                        if "Image found" in combined_context and len(context_chunks) == 1:
                            ans = "I found this image matching your request!"
                        else:
                            ans = call_llm(prompt)
                        response_content = ans
                        trust_layer["sources"] = context_chunks
                        trust_layer["explanation"] = "Retrieved from vector database via semantic search."

                # ═══════ Display results ═══════
                st.markdown(response_content)

                # Show data table INLINE (immediately visible) for structured results
                if trust_layer.get("dataframe_preview") is not None:
                    st.markdown("---")
                    st.markdown("**📊 Data Result:**")
                    try:
                        st.dataframe(trust_layer["dataframe_preview"], use_container_width=True)
                    except Exception:
                        st.dataframe(trust_layer["dataframe_preview"].astype(str), use_container_width=True)

                if trust_layer:
                    with st.expander("🔍 Source Transparency & Explainability"):
                        if trust_layer.get("intent"):
                            st.markdown(f"<span class='intent-badge'>🎯 {trust_layer['intent'].upper().replace('_', ' ')}</span>", unsafe_allow_html=True)
                        if trust_layer.get("pandas_code"):
                            st.subheader("Generated Code")
                            st.code(trust_layer["pandas_code"], language="python")
                        if trust_layer.get("explanation"):
                            st.subheader("How It Was Computed")
                            st.write(trust_layer["explanation"])
                        if trust_layer.get("sources"):
                            st.subheader("Source Chunks")
                            for idx, c in enumerate(trust_layer["sources"]):
                                st.write(f"**Chunk {idx+1}:** {c}")
                                st.divider()
                        if trust_layer.get("analysis_data"):
                            st.subheader("Raw Analysis Data")
                            st.json(trust_layer["analysis_data"])

                if chart_fig:
                    st.pyplot(chart_fig)

                if retrieved_images:
                    for img_path in retrieved_images:
                        try:
                            st.image(img_path, caption="Retrieved relevant image")
                        except:
                            pass

                # Export buttons (Feature 20)
                exp_col1, exp_col2, _ = st.columns([1, 1, 4])
                with exp_col1:
                    pdf_bytes = export_to_pdf(
                        title="Talk-to-Data Report",
                        query=query,
                        response_text=response_content,
                        chart_fig=chart_fig,
                        dataframe=trust_layer.get("dataframe_preview"),
                        trust_info=trust_layer,
                    )
                    if pdf_bytes:
                        st.download_button(
                            "📄 Export PDF", data=pdf_bytes,
                            file_name=f"insight_{datetime.now().strftime('%H%M%S')}.pdf",
                            mime="application/pdf",
                        )
                with exp_col2:
                    if trust_layer.get("dataframe_preview") is not None:
                        csv_bytes = export_dataframe_to_csv(trust_layer["dataframe_preview"])
                        if csv_bytes:
                            st.download_button(
                                "📊 Export CSV", data=csv_bytes,
                                file_name=f"data_{datetime.now().strftime('%H%M%S')}.csv",
                                mime="text/csv",
                            )

                # Cache the result (Feature 12)
                st.session_state["query_cache"][cache_key] = {
                    "content": response_content,
                    "trust_layer": {k: v for k, v in trust_layer.items() if k != "dataframe_preview"},
                    "chart": chart_fig,
                }

                # Append to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content,
                    "trust_layer": trust_layer,
                    "chart": chart_fig,
                    "retrieved_images": retrieved_images,
                    "_query": query,
                })
