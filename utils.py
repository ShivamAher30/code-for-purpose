import clip
from datetime import datetime
import faiss
import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer
import streamlit as st
import torch
import whisper

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

@st.cache_resource
def load_text_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("small")
    return model

def load_image_index():
    index = faiss.read_index('./vectorstore/image_index.index')
    data = pd.read_csv("./vectorstore/image_data.csv")
    return index, data

def load_text_index():
    index = faiss.read_index('./vectorstore/text_index.index')
    data = pd.read_csv("./vectorstore/text_data.csv")
    return index, data

def load_audio_index():
    index = faiss.read_index('./vectorstore/audio_index.index')
    data = pd.read_csv("./vectorstore/audio_data.csv")
    return index, data

def cosine_similarity(a, b):
    return torch.cosine_similarity(a, b)


def get_local_files(directory: str, extensions: list = None, get_details: bool = False):
    files = os.listdir(directory)
    if not extensions:
        if get_details:
            return [{
                "file_name": file,
                "file_size": os.path.getsize(os.path.join(directory, file)),
                "file_created": datetime.fromtimestamp(os.path.getctime(os.path.join(directory, file)))
            } for file in files]
        else:
            return files
    else:
        if get_details:
            filtered_files = []
            for file in files:
                file_extension = file.split(".")[-1]
                if file_extension in extensions:
                    filtered_files.append({
                        "file_name": file,
                        "file_size": os.path.getsize(os.path.join(directory, file)),
                        "file_created": datetime.fromtimestamp(os.path.getctime(os.path.join(directory, file)))
                    })
            return filtered_files
        else:
            return [file for file in files if file.split(".")[-1] in extensions]


# ──────────────────────────────────────────────
# FEATURE 3: Auto-Detect Schema
# ──────────────────────────────────────────────

def auto_detect_schema(df):
    """
    Automatically detect column types, date fields, and data characteristics.
    Returns a structured schema dict.
    """
    schema = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": {},
        "date_columns": [],
        "numeric_columns": [],
        "categorical_columns": [],
        "text_columns": [],
    }

    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().sum() / len(df) * 100, 1) if len(df) > 0 else 0,
            "unique_count": int(df[col].nunique()),
            "sample_values": [str(v) for v in df[col].dropna().head(3).tolist()],
        }

        # Try to detect dates
        if df[col].dtype == "object":
            sample = df[col].dropna().head(30)
            if len(sample) > 0:
                try:
                    parsed = pd.to_datetime(sample, infer_datetime_format=True)
                    col_info["detected_type"] = "datetime"
                    schema["date_columns"].append(col)
                except (ValueError, TypeError):
                    # Check if it's a category or free text
                    if df[col].nunique() < min(50, len(df) * 0.3):
                        col_info["detected_type"] = "categorical"
                        schema["categorical_columns"].append(col)
                    else:
                        col_info["detected_type"] = "text"
                        schema["text_columns"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_info["detected_type"] = "numeric"
            col_info["min"] = round(float(df[col].min()), 2) if not df[col].isnull().all() else None
            col_info["max"] = round(float(df[col].max()), 2) if not df[col].isnull().all() else None
            col_info["mean"] = round(float(df[col].mean()), 2) if not df[col].isnull().all() else None
            schema["numeric_columns"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info["detected_type"] = "datetime"
            schema["date_columns"].append(col)
        else:
            col_info["detected_type"] = "other"

        schema["columns"][col] = col_info

    return schema


# ──────────────────────────────────────────────
# FEATURE 7: PII / Sensitive Field Detection & Masking
# ──────────────────────────────────────────────

# Patterns for common PII column names
_PII_PATTERNS = [
    r"ssn", r"social.?security", r"email", r"e.?mail", r"phone", r"telephone",
    r"mobile", r"address", r"street", r"zip.?code", r"postal", r"credit.?card",
    r"card.?number", r"cvv", r"password", r"passport", r"driver.?license",
    r"birth.?date", r"dob", r"date.?of.?birth", r"national.?id", r"tax.?id",
    r"bank.?account", r"routing.?number", r"iban", r"swift",
    r"first.?name", r"last.?name", r"full.?name", r"contact",
]


def detect_sensitive_columns(df):
    """
    Detect columns likely containing PII or sensitive data.
    Returns list of (column_name, reason) tuples.
    """
    sensitive = []
    for col in df.columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        for pattern in _PII_PATTERNS:
            if re.search(pattern, col_lower):
                sensitive.append((col, f"Column name matches PII pattern '{pattern}'"))
                break
        else:
            # Value-level pattern detection (sample check)
            sample = df[col].dropna().astype(str).head(20)
            if len(sample) == 0:
                continue
            # Email pattern
            email_matches = sample.str.match(r"^[^@]+@[^@]+\.[^@]+$").sum()
            if email_matches > len(sample) * 0.5:
                sensitive.append((col, "Values appear to be email addresses"))
                continue
            # Phone pattern (rough)
            phone_matches = sample.str.match(r"^[\+\d\s\-\(\)]{7,15}$").sum()
            if phone_matches > len(sample) * 0.7:
                # Also ensure it's not just normal numbers
                if df[col].dtype == "object":
                    sensitive.append((col, "Values appear to be phone numbers"))

    return sensitive


def mask_sensitive_data(df, sensitive_cols):
    """
    Mask PII columns by replacing values with '****'.
    Returns a new DataFrame — does not modify the original.
    """
    df_masked = df.copy()
    for col, _ in sensitive_cols:
        if col in df_masked.columns:
            df_masked[col] = "****"
    return df_masked
