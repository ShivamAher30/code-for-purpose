"""
api_server.py — FastAPI Backend for Talk-to-Data React Frontend
Wraps all existing Python modules into REST endpoints.
"""

import os
import io
import json
import hashlib
import base64
import tempfile
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ── Import existing modules ──
from utils import (
    load_clip_model, load_text_embedding_model, load_whisper_model,
    load_text_index, load_audio_index, load_image_index,
    auto_detect_schema, detect_sensitive_columns, mask_sensitive_data,
)
from vectordb import (
    search_text_index, search_image_index,
    add_audio_to_index, add_pdf_to_index, add_image_to_index,
    update_vectordb,
)
from data_upload.input_sources_utils.text_util import process_text
from llm_engine import (
    call_llm, route_intent, nl_to_pandas, nl_to_pandas_with_retry,
    generate_explanation, generate_insights, generate_answer_from_pandas,
    check_query_clarity, narrate_comparison, narrate_root_cause,
    narrate_breakdown, narrate_summary, narrate_anomalies,
    detect_metric_and_dimensions, analyze_vision_chart,
)
from visualize import (
    execute_pandas_code_safely, generate_auto_chart, get_auto_explanation,
)
from analysis_engine import (
    root_cause_analysis, compare_periods, compare_segments,
    breakdown_metric, detect_anomalies, generate_data_summary, rank_insights,
)
from export_engine import export_to_pdf, export_dataframe_to_csv
from visualize import generate_auto_chart

# ── App Setup ──
app = FastAPI(title="Talk-to-Data API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_clip_model()
text_embedding_model = load_text_embedding_model()
whisper_model = load_whisper_model()

Path("./vectorstore").mkdir(parents=True, exist_ok=True)
Path("./images").mkdir(parents=True, exist_ok=True)
Path("./audio").mkdir(parents=True, exist_ok=True)

# Mount static file directories for serving uploaded files
try:
    app.mount("/files/images", StaticFiles(directory="images"), name="images")
    app.mount("/files/audio", StaticFiles(directory="audio"), name="audio")
except Exception:
    pass

# In-memory state (per session — for prototype)
state = {
    "df": None,
    "schema": None,
    "sensitive_cols": [],
    "mask_sensitive": False,
    "messages": [],
    "query_cache": {},
}


# ── Pydantic Models ──
class QueryRequest(BaseModel):
    query: str
    routing_mode: str = "Auto-Detect"


class QuickActionRequest(BaseModel):
    action: str  # "summary", "anomaly", "insights"


class MetricRequest(BaseModel):
    name: str
    definition: str


class ChartRequest(BaseModel):
    query: str
    chart_type: Optional[str] = None  # bar, line, pie, scatter, etc.


# ── Helper: DataFrame to chart-friendly JSON ──
def _df_to_chart_data(result_data, query_hint=""):
    """Convert pandas result to JSON-friendly chart data for Recharts."""
    if result_data is None:
        return None, None

    if not isinstance(result_data, (pd.DataFrame, pd.Series)):
        return None, None

    if isinstance(result_data, pd.Series):
        if len(result_data) < 2:
            return None, None
        chart_data = [
            {"name": str(idx), "value": float(val) if pd.notna(val) else 0}
            for idx, val in result_data.items()
        ]
        # Detect chart type
        n = len(result_data)
        query_lower = query_hint.lower()
        if any(k in query_lower for k in ["pie", "breakdown", "share", "proportion"]):
            chart_type = "pie"
        elif any(k in query_lower for k in ["trend", "over time", "monthly", "daily"]):
            chart_type = "line"
        elif n <= 7 and (result_data > 0).all():
            chart_type = "pie"
        elif n <= 20:
            chart_type = "bar"
        else:
            chart_type = "line"
        return chart_data, chart_type

    elif isinstance(result_data, pd.DataFrame):
        if result_data.empty:
            return None, None
        numeric_cols = result_data.select_dtypes(include="number").columns.tolist()
        non_numeric = [c for c in result_data.columns if c not in numeric_cols]
        label_col = non_numeric[0] if non_numeric else result_data.index.name or "index"

        chart_data = []
        for idx, row in result_data.iterrows():
            entry = {"name": str(row[label_col]) if label_col in result_data.columns else str(idx)}
            for col in numeric_cols:
                entry[col] = float(row[col]) if pd.notna(row[col]) else 0
            chart_data.append(entry)

        n_rows = len(result_data)
        n_numeric = len(numeric_cols)
        query_lower = query_hint.lower()

        if any(k in query_lower for k in ["pie", "breakdown", "share"]):
            chart_type = "pie"
        elif any(k in query_lower for k in ["trend", "over time", "monthly"]):
            chart_type = "line"
        elif n_numeric >= 2 and n_rows <= 15:
            chart_type = "grouped_bar"
        elif n_rows <= 20:
            chart_type = "bar"
        else:
            chart_type = "line"
        return chart_data, chart_type

    return None, None


def _df_to_table(result_data):
    """Convert pandas result to table-friendly JSON."""
    if result_data is None:
        return None

    if isinstance(result_data, pd.Series):
        result_data = result_data.reset_index()
        result_data.columns = [str(c) for c in result_data.columns]

    if isinstance(result_data, pd.DataFrame):
        # Limit to 100 rows for frontend
        display_df = result_data.head(100)
        return {
            "columns": [str(c) for c in display_df.columns],
            "rows": display_df.astype(str).values.tolist(),
            "total_rows": len(result_data),
            "shown_rows": len(display_df),
        }
    return None


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {"status": "ok", "has_data": state["df"] is not None}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), sheet_name: Optional[str] = Form(None)):
    """Upload CSV or Excel file."""
    try:
        contents = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            state["df"] = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(io.BytesIO(contents))
            sheets = xls.sheet_names
            selected = sheet_name if sheet_name and sheet_name in sheets else sheets[0]
            state["df"] = pd.read_excel(io.BytesIO(contents), sheet_name=selected)
            if len(sheets) > 1:
                return JSONResponse({
                    "success": True,
                    "rows": len(state["df"]),
                    "columns": len(state["df"].columns),
                    "sheets": sheets,
                    "selected_sheet": selected,
                    "schema": _build_schema_response(),
                })
        else:
            raise HTTPException(400, "Unsupported file type. Use CSV or Excel.")

        state["schema"] = auto_detect_schema(state["df"])
        state["sensitive_cols"] = detect_sensitive_columns(state["df"])
        state["messages"] = []
        state["query_cache"] = {}

        return {
            "success": True,
            "rows": len(state["df"]),
            "columns": len(state["df"].columns),
            "schema": _build_schema_response(),
        }
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


def _build_schema_response():
    """Build schema response for frontend."""
    if not state["schema"]:
        return None
    schema = state["schema"]
    return {
        "total_rows": schema["total_rows"],
        "total_columns": schema["total_columns"],
        "numeric_columns": schema.get("numeric_columns", []),
        "categorical_columns": schema.get("categorical_columns", []),
        "date_columns": schema.get("date_columns", []),
        "text_columns": schema.get("text_columns", []),
        "columns": schema.get("columns", {}),
        "sensitive_columns": [
            {"name": col, "reason": reason}
            for col, reason in state["sensitive_cols"]
        ],
    }


@app.get("/api/schema")
def get_schema():
    """Return the current dataset schema."""
    if state["df"] is None:
        return {"has_data": False}
    return {
        "has_data": True,
        "schema": _build_schema_response(),
        "preview": state["df"].head(5).astype(str).to_dict("records"),
    }


@app.post("/api/query")
async def process_query(req: QueryRequest):
    """Process a natural language query against the data."""
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query cannot be empty.")

    has_df = state["df"] is not None
    has_rag_text = os.path.exists("./vectorstore/text_index.index")
    has_rag_audio = os.path.exists("./vectorstore/audio_index.index")
    has_rag_image = os.path.exists("./vectorstore/image_index.index")
    has_rag = has_rag_text or has_rag_audio or has_rag_image

    if not has_df and not has_rag:
        return {
            "response": "👋 Welcome! Please upload some data (CSV or Excel) to get started.",
            "intent": "none",
        }

    # Intent routing
    if req.routing_mode == "Structured (CSV)":
        intent = "structured"
    elif req.routing_mode == "Unstructured (RAG)":
        intent = "rag"
    else:
        intent = route_intent(query, has_df, has_rag)

    # Get working DataFrame
    df = state["df"]
    if df is not None and state["mask_sensitive"] and state["sensitive_cols"]:
        df_for_llm = mask_sensitive_data(df, state["sensitive_cols"])
    else:
        df_for_llm = df

    # Cache check
    cache_key = hashlib.md5(f"{query}_{intent}".encode()).hexdigest()
    cached = state["query_cache"].get(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    result = {
        "intent": intent,
        "response": "",
        "chart_data": None,
        "chart_type": None,
        "chart_keys": None,
        "table_data": None,
        "trust_layer": {"intent": intent},
        "cached": False,
    }

    try:
        # ── STRUCTURED ──
        if intent == "structured":
            if not has_df:
                result["response"] = "No CSV/Excel loaded. Please upload data first."
                return result

            is_clear, clarification_q = check_query_clarity(query, df_for_llm)
            if not is_clear:
                result["response"] = f"🤔 {clarification_q}"
                result["trust_layer"]["explanation"] = "Query was ambiguous — requested clarification."
                return result

            history = state["messages"][-10:]
            code, df_result, err = nl_to_pandas_with_retry(query, df_for_llm, history=history)
            result["trust_layer"]["pandas_code"] = code

            if err:
                result["response"] = f"I encountered an issue processing that query. Error: {err}\n\nTry rephrasing your question."
                result["trust_layer"]["explanation"] = f"Code execution failed: {err}"
            else:
                chart_data, chart_type = _df_to_chart_data(df_result, query)
                result["chart_data"] = chart_data
                result["chart_type"] = chart_type
                if isinstance(df_result, pd.DataFrame) and len(df_result.select_dtypes(include="number").columns) > 1:
                    result["chart_keys"] = df_result.select_dtypes(include="number").columns.tolist()

                explanation = generate_explanation(query, None, df_result)
                result["trust_layer"]["explanation"] = explanation
                result["table_data"] = _df_to_table(df_result)
                result["response"] = generate_answer_from_pandas(query, df_result)

        # ── COMPARISON ──
        elif intent == "comparison":
            if not has_df:
                result["response"] = "No data uploaded for comparison."
                return result

            metric_col, dim_cols, period_col = detect_metric_and_dimensions(query, df_for_llm)

            if metric_col and period_col:
                unique_periods = df_for_llm[period_col].dropna().unique()
                if len(unique_periods) >= 2:
                    comp = compare_periods(df_for_llm, metric_col, period_col, unique_periods[-2], unique_periods[-1])
                    result["response"] = narrate_comparison(query, comp)
                    # Build comparison chart data
                    pa, pb = comp["period_a"], comp["period_b"]
                    result["chart_data"] = [
                        {"name": "Sum", str(pa["period"]): pa["sum"], str(pb["period"]): pb["sum"]},
                        {"name": "Average", str(pa["period"]): pa["mean"], str(pb["period"]): pb["mean"]},
                        {"name": "Count", str(pa["period"]): pa["count"], str(pb["period"]): pb["count"]},
                    ]
                    result["chart_type"] = "grouped_bar"
                    result["chart_keys"] = [str(pa["period"]), str(pb["period"])]
                    result["trust_layer"]["analysis_data"] = comp
                    result["trust_layer"]["explanation"] = f"Compared '{metric_col}' across '{period_col}'"
                else:
                    result["response"] = f"Only one period found in '{period_col}'. Need at least two periods."
            elif metric_col and dim_cols:
                comp = compare_segments(df_for_llm, metric_col, dim_cols[0])
                result["response"] = narrate_breakdown(query, {
                    "metric": metric_col,
                    "grand_total": comp["grand_total"],
                    "breakdowns": {dim_cols[0]: {"segments": comp["segments"]}}
                })
                seg_df = pd.DataFrame(comp["segments"])
                if not seg_df.empty:
                    chart_data, chart_type = _df_to_chart_data(seg_df.set_index(dim_cols[0])["total"], query)
                    result["chart_data"] = chart_data
                    result["chart_type"] = chart_type
                result["trust_layer"]["analysis_data"] = comp
            else:
                # Fallback to structured
                history = state["messages"][-10:]
                code, df_result, err = nl_to_pandas_with_retry(query, df_for_llm, history=history)
                result["trust_layer"]["pandas_code"] = code
                if err:
                    result["response"] = f"Could not execute comparison: {err}"
                else:
                    chart_data, chart_type = _df_to_chart_data(df_result, query)
                    result["chart_data"] = chart_data
                    result["chart_type"] = chart_type
                    result["table_data"] = _df_to_table(df_result)
                    result["response"] = generate_answer_from_pandas(query, df_result)

        # ── ROOT CAUSE ──
        elif intent == "root_cause":
            if not has_df:
                result["response"] = "No data uploaded for root cause analysis."
                return result

            metric_col, dim_cols, period_col = detect_metric_and_dimensions(query, df_for_llm)
            if metric_col:
                rca = root_cause_analysis(df_for_llm, metric_col, period_col=period_col)
                result["response"] = narrate_root_cause(query, rca)
                result["trust_layer"]["analysis_data"] = rca
                if "top_drivers" in rca:
                    drivers = rca["top_drivers"][:10]
                    result["chart_data"] = [
                        {
                            "name": f"{d['dimension']}: {d['segment']}",
                            "value": d["contribution_pct"],
                            "change": d["change"],
                        }
                        for d in drivers
                    ]
                    result["chart_type"] = "horizontal_bar"
                result["trust_layer"]["explanation"] = f"Root cause analysis on '{metric_col}'"
            else:
                result["response"] = "I couldn't identify the metric to analyze."

        # ── BREAKDOWN ──
        elif intent == "breakdown":
            if not has_df:
                result["response"] = "No data uploaded."
                return result

            metric_col, dim_cols, period_col = detect_metric_and_dimensions(query, df_for_llm)
            if metric_col:
                bd = breakdown_metric(df_for_llm, metric_col, group_cols=dim_cols or None)
                result["response"] = narrate_breakdown(query, bd)
                result["trust_layer"]["analysis_data"] = bd
                if bd.get("breakdowns"):
                    first_dim = list(bd["breakdowns"].keys())[0]
                    segs = bd["breakdowns"][first_dim]["segments"]
                    result["chart_data"] = [
                        {"name": str(s.get(first_dim, "")), "value": s.get("total", 0), "share": s.get("share_pct", 0)}
                        for s in segs[:15]
                    ]
                    result["chart_type"] = "pie" if len(segs) <= 8 else "bar"
                result["trust_layer"]["explanation"] = f"Breakdown of '{metric_col}'"
            else:
                result["response"] = "I couldn't identify the metric to break down."

        # ── SUMMARY ──
        elif intent == "summary":
            if not has_df:
                result["response"] = "No data uploaded."
                return result
            summary = generate_data_summary(df_for_llm)
            result["response"] = narrate_summary(query, summary)
            result["trust_layer"]["analysis_data"] = summary
            result["trust_layer"]["explanation"] = "Comprehensive analysis covering all columns."

        # ── ANOMALY ──
        elif intent == "anomaly":
            if not has_df:
                result["response"] = "No data uploaded."
                return result
            anomalies = detect_anomalies(df_for_llm)
            result["response"] = narrate_anomalies(query, anomalies)
            result["trust_layer"]["analysis_data"] = anomalies
            if anomalies.get("has_anomalies"):
                first_col = list(anomalies["anomalies"].keys())[0]
                info = anomalies["anomalies"][first_col]
                # Build scatter-like data for anomaly visualization
                col_data = df_for_llm[first_col].values
                chart_data = []
                anomaly_indices = set(info.get("indices", []))
                for i, val in enumerate(col_data[:200]):  # Limit to 200 points
                    entry = {"index": i, "value": float(val) if pd.notna(val) else 0}
                    entry["is_anomaly"] = i in anomaly_indices
                    chart_data.append(entry)
                result["chart_data"] = chart_data
                result["chart_type"] = "anomaly_scatter"
                result["trust_layer"]["bounds"] = {
                    "upper": info["upper_bound"],
                    "lower": info["lower_bound"],
                    "column": first_col,
                }
            result["trust_layer"]["explanation"] = f"IQR-based anomaly detection"

        # ── RAG ──
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

            if not context_chunks:
                result["response"] = "I couldn't find relevant information. Try uploading more data or rephrasing."
            else:
                combined_context = " ".join(context_chunks)[:1500]
                prompt = f"Answer the question based on the context below.\n\nContext: {combined_context}\n\nQuestion: {query}\n\nAnswer:"
                result["response"] = call_llm(prompt)
                result["trust_layer"]["sources"] = context_chunks
                result["trust_layer"]["explanation"] = "Retrieved from vector database via semantic search."

    except Exception as e:
        result["response"] = f"An error occurred: {str(e)}"
        result["trust_layer"]["explanation"] = str(e)

    # Cache result
    state["query_cache"][cache_key] = {k: v for k, v in result.items()}

    # Add to message history
    state["messages"].append({"role": "user", "content": query})
    state["messages"].append({"role": "assistant", "content": result["response"]})

    return result


@app.post("/api/quick-action")
def quick_action(req: QuickActionRequest):
    """Execute a quick action (summary, anomaly, insights)."""
    df = state["df"]
    if df is None:
        raise HTTPException(400, "No data uploaded.")

    if req.action == "summary":
        summary = generate_data_summary(df)
        narrative = narrate_summary("Give me a summary of this data", summary)
        return {
            "response": narrative,
            "intent": "summary",
            "trust_layer": {"intent": "summary", "analysis_data": summary},
        }

    elif req.action == "anomaly":
        anomalies = detect_anomalies(df)
        narrative = narrate_anomalies("Detect anomalies in this data", anomalies)
        chart_data = None
        chart_type = None
        bounds = None
        if anomalies.get("has_anomalies"):
            first_col = list(anomalies["anomalies"].keys())[0]
            info = anomalies["anomalies"][first_col]
            col_data = df[first_col].values
            chart_data = []
            anomaly_indices = set(info.get("indices", []))
            for i, val in enumerate(col_data[:200]):
                entry = {"index": i, "value": float(val) if pd.notna(val) else 0}
                entry["is_anomaly"] = i in anomaly_indices
                chart_data.append(entry)
            chart_type = "anomaly_scatter"
            bounds = {"upper": info["upper_bound"], "lower": info["lower_bound"], "column": first_col}
        return {
            "response": narrative,
            "intent": "anomaly",
            "chart_data": chart_data,
            "chart_type": chart_type,
            "trust_layer": {"intent": "anomaly", "analysis_data": anomalies, "bounds": bounds},
        }

    elif req.action == "insights":
        insights = generate_insights(df)
        return {
            "response": insights,
            "intent": "insights",
            "trust_layer": {"intent": "structured"},
        }

    raise HTTPException(400, f"Unknown action: {req.action}")


@app.post("/api/export/csv")
def export_csv():
    """Export current data result as CSV."""
    if state["df"] is None:
        raise HTTPException(400, "No data to export.")
    csv_bytes = export_dataframe_to_csv(state["df"])
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=data_{datetime.now().strftime('%H%M%S')}.csv"}
    )


@app.get("/api/metrics")
def get_metrics():
    """Get all defined business metrics."""
    dict_path = "semantic_dict.json"
    if os.path.exists(dict_path):
        try:
            with open(dict_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


@app.post("/api/metrics")
def add_metric(req: MetricRequest):
    """Add a new business metric definition."""
    dict_path = "semantic_dict.json"
    metrics = {}
    if os.path.exists(dict_path):
        try:
            with open(dict_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    metrics[req.name] = req.definition
    with open(dict_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return {"success": True, "metrics": metrics}


@app.post("/api/settings/privacy")
def toggle_privacy(enabled: bool = Form(False)):
    """Toggle PII masking."""
    state["mask_sensitive"] = enabled
    return {"mask_sensitive": state["mask_sensitive"]}


@app.post("/api/clear")
def clear_chat():
    """Clear chat history and cache."""
    state["messages"] = []
    state["query_cache"] = {}
    return {"success": True}


@app.post("/api/clear-rag")
def clear_rag_data():
    """Clear all RAG vector indices, metadata, and uploaded files."""
    removed = {"indices": [], "metadata": [], "audio": 0, "images": 0}

    # Remove FAISS indices and metadata CSVs
    index_files = [
        "text_index.index", "audio_index.index", "image_index.index",
    ]
    metadata_files = [
        "text_data.csv", "audio_data.csv", "image_data.csv",
    ]

    for fname in index_files:
        fpath = f"./vectorstore/{fname}"
        if os.path.exists(fpath):
            os.remove(fpath)
            removed["indices"].append(fname)

    for fname in metadata_files:
        fpath = f"./vectorstore/{fname}"
        if os.path.exists(fpath):
            os.remove(fpath)
            removed["metadata"].append(fname)

    # Clear uploaded audio files
    if os.path.exists("./audio"):
        for f in os.listdir("./audio"):
            try:
                os.remove(os.path.join("./audio", f))
                removed["audio"] += 1
            except Exception:
                pass

    # Clear uploaded images
    if os.path.exists("./images"):
        for f in os.listdir("./images"):
            try:
                os.remove(os.path.join("./images", f))
                removed["images"] += 1
            except Exception:
                pass

    return {"success": True, "removed": removed}


# ══════════════════════════════════════════════
# RAG FILE UPLOAD ENDPOINTS (Audio, PDF, Image, Text)
# ══════════════════════════════════════════════

@app.post("/api/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and process an audio file — transcribes via Whisper and indexes."""
    try:
        contents = await file.read()
        filename = file.filename.replace(" ", "_")
        audio_path = f"./audio/{filename}"
        with open(audio_path, "wb") as f:
            f.write(contents)

        # Transcribe
        transcript = whisper_model.transcribe(audio_path)["text"]

        # Chunk and index
        from langchain_text_splitters import CharacterTextSplitter
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200,
            length_function=len, is_separator_regex=False,
        )
        chunks = text_splitter.split_text(transcript)
        text_embeddings = text_embedding_model.encode(chunks)
        for i, chunk in enumerate(chunks):
            from vectordb import update_vectordb as _update_vdb
            _update_vdb(
                index_path="audio_index.index",
                embedding=text_embeddings[i],
                text_content=chunk,
                audio_path=audio_path,
            )

        return {
            "success": True,
            "filename": filename,
            "transcript_preview": transcript[:300],
            "chunks_indexed": len(chunks),
            "type": "audio",
        }
    except Exception as e:
        raise HTTPException(500, f"Audio processing failed: {str(e)}")


@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF — extracts text + images, indexes into vector DB."""
    try:
        contents = await file.read()
        pdf_bytes = io.BytesIO(contents)
        pdf_bytes.name = file.filename

        from PyPDF2 import PdfReader
        from langchain_text_splitters import CharacterTextSplitter

        reader = PdfReader(pdf_bytes)
        total_chunks = 0
        total_images = 0
        all_text = []

        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200,
            length_function=len, is_separator_regex=False,
        )

        for page_num, page in enumerate(reader.pages):
            # Extract images
            try:
                for img in page.images:
                    img.name = f"{page_num}_{img.name}"
                    add_image_to_index(img, clip_model, preprocess)
                    total_images += 1
            except Exception:
                pass

            # Extract text
            page_text = page.extract_text() or ""
            all_text.append(page_text)
            if page_text.strip():
                chunks = text_splitter.split_text(page_text)
                text_embeddings = text_embedding_model.encode(chunks)
                for i, chunk in enumerate(chunks):
                    from vectordb import update_vectordb as _update_vdb
                    _update_vdb(
                        index_path="text_index.index",
                        embedding=text_embeddings[i],
                        text_content=chunk,
                    )
                    total_chunks += 1

        return {
            "success": True,
            "filename": file.filename,
            "pages": len(reader.pages),
            "chunks_indexed": total_chunks,
            "images_indexed": total_images,
            "text_preview": " ".join(all_text)[:300],
            "type": "pdf",
        }
    except Exception as e:
        raise HTTPException(500, f"PDF processing failed: {str(e)}")


@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload and process an image — indexes via CLIP."""
    try:
        contents = await file.read()
        img_bytes = io.BytesIO(contents)
        img_bytes.name = file.filename

        add_image_to_index(img_bytes, clip_model, preprocess)

        return {
            "success": True,
            "filename": file.filename,
            "type": "image",
        }
    except Exception as e:
        raise HTTPException(500, f"Image processing failed: {str(e)}")


@app.post("/api/upload/text")
async def upload_text(file: UploadFile = File(...)):
    """Upload and process a plain text file."""
    try:
        contents = await file.read()
        text = contents.decode("utf-8", errors="ignore")

        process_text(text, text_embedding_model)

        return {
            "success": True,
            "filename": file.filename,
            "text_preview": text[:300],
            "type": "text",
        }
    except Exception as e:
        raise HTTPException(500, f"Text processing failed: {str(e)}")


@app.get("/api/rag-files")
def list_rag_files():
    """List all uploaded RAG files (audio, images, text/PDF indices)."""
    files = {
        "audio": [],
        "images": [],
        "has_text_index": os.path.exists("./vectorstore/text_index.index"),
        "has_audio_index": os.path.exists("./vectorstore/audio_index.index"),
        "has_image_index": os.path.exists("./vectorstore/image_index.index"),
    }

    if os.path.exists("./audio"):
        files["audio"] = [
            {"name": f, "path": f"/files/audio/{f}"}
            for f in os.listdir("./audio")
            if f.endswith((".mp3", ".wav", ".m4a", ".ogg", ".webm"))
        ]

    if os.path.exists("./images"):
        files["images"] = [
            {"name": f, "path": f"/files/images/{f}"}
            for f in os.listdir("./images")
            if f.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
        ]

    return files


@app.post("/api/generate-chart")
async def generate_chart(req: ChartRequest):
    """Generate a professional chart from a natural language query against loaded data."""
    if state["df"] is None:
        raise HTTPException(400, "No data loaded. Upload a CSV/Excel file first.")

    df = state["df"]
    query = req.query.strip()

    try:
        # Use LLM to generate pandas code for the chart data
        history = state["messages"][-10:]
        code, df_result, err = nl_to_pandas_with_retry(query, df, history=history)

        if err:
            return {
                "success": False,
                "error": err,
                "response": f"Could not generate chart: {err}",
            }

        # Convert to chart data for Recharts
        chart_data, chart_type_auto = _df_to_chart_data(df_result, query)

        # Override chart type if user specified
        chart_type = req.chart_type or chart_type_auto

        # Also get chart keys for multi-series
        chart_keys = None
        if isinstance(df_result, pd.DataFrame) and len(df_result.select_dtypes(include="number").columns) > 1:
            chart_keys = df_result.select_dtypes(include="number").columns.tolist()

        # Generate explanation
        explanation = generate_answer_from_pandas(query, df_result)

        return {
            "success": True,
            "chart_data": chart_data,
            "chart_type": chart_type,
            "chart_keys": chart_keys,
            "table_data": _df_to_table(df_result),
            "response": explanation,
            "pandas_code": code,
        }
    except Exception as e:
        raise HTTPException(500, f"Chart generation failed: {str(e)}")


@app.post("/api/export/pdf")
async def export_pdf_endpoint(query: str = Form(""), response_text: str = Form("")):
    """Export a query result as a PDF report."""
    pdf_bytes = export_to_pdf(
        title="Talk-to-Data Report",
        query=query,
        response_text=response_text,
        dataframe=state["df"],
    )
    if pdf_bytes is None:
        raise HTTPException(500, "PDF generation failed. reportlab may not be installed.")
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{datetime.now().strftime('%H%M%S')}.pdf"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
