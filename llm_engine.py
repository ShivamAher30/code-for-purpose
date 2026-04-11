"""
llm_engine.py — Core Intelligence Module for Talk-to-Data
Handles: Intent routing, NL→Pandas conversion, error retry, clarification engine,
         comparison prompts, root cause narration, insight generation.
Multi-agent architecture: Each function acts as a specialized "agent" with its own prompt.
"""

import pandas as pd
import re
import os
import json
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

MODEL = "llama-3.3-70b-versatile"

# ──────────────────────────────────────────────
# BASE LLM CALL
# ──────────────────────────────────────────────

def call_llm(prompt, model=MODEL, system_message=None, temperature=0.0):
    """
    Calls the Groq API with optional system message (multi-agent support).
    """
    try:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"Error connecting to LLM: {e}"


# ──────────────────────────────────────────────
# AGENT 1: INTENT ROUTER (Feature 19 — Multi-Agent)
# ──────────────────────────────────────────────

INTENT_SYSTEM = """You are an intent classification agent for a data analytics assistant.
Classify user queries into EXACTLY one of these categories:
- structured: Questions about CSV/tabular data (calculations, aggregations, filtering, sorting)
- rag: Questions about uploaded documents, PDFs, websites, audio transcripts
- vision: Questions analyzing uploaded images/charts
- comparison: Questions comparing two periods, segments, or groups (contains "vs", "compared to", "difference between")
- root_cause: Questions asking WHY something changed (contains "why", "what caused", "reason for", "dropped", "increased")
- breakdown: Questions asking for decomposition (contains "break down", "by region", "by category", "distribution across")
- summary: Questions asking for overall summary or overview (contains "summarize", "overview", "summary", "what's happening")
- anomaly: Questions about outliers or unusual patterns (contains "anomaly", "unusual", "outlier", "spike", "sudden")

Reply with ONLY the intent label, nothing else."""

STRUCTURED_KEYWORDS = [
    "average", "sum", "count", "mean", "median", "plot", "chart", "trend",
    "csv", "data", "table", "dataframe", "columns", "sort", "filter",
    "group by", "histogram", "bar", "line", "top", "bottom", "highest",
    "lowest", "total", "maximum", "minimum", "revenue", "sales", "profit",
    "cost", "price", "show me", "list", "find", "how many", "what is",
]

RAG_KEYWORDS = [
    "document", "pdf", "image", "audio", "website", "text", "chunk",
    "context", "article", "page", "paragraph", "resume",
]

COMPARISON_KEYWORDS = [
    "vs", "versus", "compared to", "comparison", "compare", "difference between",
    "this week vs", "this month vs", "last month", "last week", "last year",
    "year over year", "month over month", "week over week",
]

ROOT_CAUSE_KEYWORDS = [
    "why did", "why is", "what caused", "reason for", "root cause",
    "dropped", "decreased", "increased", "spiked", "declined", "fell",
    "went up", "went down", "change in", "driver",
]

BREAKDOWN_KEYWORDS = [
    "break down", "breakdown", "decompose", "by region", "by category",
    "by product", "by segment", "by channel", "distribution across",
    "split by", "grouped by", "per region", "per category",
]

SUMMARY_KEYWORDS = [
    "summarize", "summary", "overview", "what's happening", "key findings",
    "highlights", "status", "report", "brief",
]

ANOMALY_KEYWORDS = [
    "anomaly", "anomalies", "unusual", "outlier", "spike", "sudden",
    "unexpected", "abnormal", "weird", "strange",
]


def route_intent(query, has_df=False, has_rag=False):
    """
    Classifies intent of query using keyword scoring + LLM fallback.
    Returns one of: structured, rag, vision, comparison, root_cause, breakdown, summary, anomaly
    """
    query_lower = query.lower()

    # Score each intent
    scores = {
        "structured": sum(1 for k in STRUCTURED_KEYWORDS if k in query_lower),
        "rag": sum(1 for k in RAG_KEYWORDS if k in query_lower),
        "comparison": sum(2 for k in COMPARISON_KEYWORDS if k in query_lower),
        "root_cause": sum(2 for k in ROOT_CAUSE_KEYWORDS if k in query_lower),
        "breakdown": sum(2 for k in BREAKDOWN_KEYWORDS if k in query_lower),
        "summary": sum(2 for k in SUMMARY_KEYWORDS if k in query_lower),
        "anomaly": sum(2 for k in ANOMALY_KEYWORDS if k in query_lower),
    }

    # Only DF available — route structured intents
    if has_df and not has_rag:
        max_intent = max(scores, key=scores.get)
        if scores[max_intent] > 0 and max_intent not in ("rag",):
            return max_intent
        return "structured"

    # Only RAG available
    if has_rag and not has_df:
        return "rag"

    # Both available — find top scoring
    max_intent = max(scores, key=scores.get)
    if scores[max_intent] >= 2:
        return max_intent

    # Ambiguous — ask LLM
    if has_df and has_rag:
        ans = call_llm(query, system_message=INTENT_SYSTEM).lower().strip()
        valid_intents = ["structured", "rag", "comparison", "root_cause", "breakdown", "summary", "anomaly"]
        for intent in valid_intents:
            if intent in ans:
                return intent
        return "structured"

    return "structured" if has_df else "rag"


# ──────────────────────────────────────────────
# AGENT 2: CLARIFICATION ENGINE (Feature 15)
# ──────────────────────────────────────────────

def check_query_clarity(query, df):
    """
    Lightweight check — only reject truly empty/meaningless queries.
    Almost everything should pass through and let the LLM try to handle it.
    The LLM is smart enough to figure out intent from context.
    Returns: (is_clear: bool, clarification_question: str or None)
    """
    if df is None:
        return True, None

    query_stripped = query.strip()

    # Only reject if query is essentially empty
    if len(query_stripped) < 3:
        return False, "Could you please provide a more detailed question about your data?"

    # Only reject truly meaningless single-word queries
    meaningless = {"hi", "hello", "hey", "help", "test", "ok", "yes", "no", "thanks"}
    if query_stripped.lower() in meaningless:
        columns = list(df.columns)
        return False, f"I'm ready to help! Your dataset has these columns: **{', '.join(columns[:10])}**. What would you like to know about your data?"

    # Everything else — let the LLM handle it
    return True, None


# ──────────────────────────────────────────────
# AGENT 3: DATA EXECUTOR (NL→Pandas) (Feature 2)
# ──────────────────────────────────────────────

def load_semantic_dict():
    try:
        if os.path.exists("semantic_dict.json"):
            with open("semantic_dict.json", "r") as f:
                return json.load(f)
    except:
        pass
    return None


def nl_to_pandas(query, df, history=None):
    """
    Translates Natural Language to Pandas code using the LLM.
    Returns python code string.
    """
    columns = list(df.columns)
    sample_data = df.head(3).to_dict()
    dtypes = {col: str(df[col].dtype) for col in df.columns}

    history_str = ""
    if history:
        # Include recent relevant history for conversational context
        recent = history[-8:] if len(history) >= 8 else history
        history_str = "Chat History:\n" + "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent if 'content' in msg]
        )

    semantic_dict = load_semantic_dict()
    semantic_str = ""
    if semantic_dict:
        semantic_str = "\nStrict Business Metric Definitions (USE THESE EXACT MATHEMATICAL RULES IF ASKED):\n" + json.dumps(semantic_dict, indent=2) + "\n"

    system_msg = """You are an expert data analyst that converts natural language questions into Pandas code.

Your process:
1. FIRST understand what the user is really asking — even if the query is vague or informal
2. Map the user's intent to the most relevant columns in the dataset
3. Generate exactly ONE valid Python Pandas expression

Rules:
- The dataframe is named `df`
- Return ONLY Python code inside a ```python block — NO explanation text
- The code must produce a pandas Series, DataFrame, or scalar
- Allowed operations: groupby, sum, mean, count, sort_values, sort_index, filtering (df[df['col'] > val]), value_counts, nlargest, nsmallest, basic math, pd.to_datetime, head, tail, describe, nunique, agg, pivot_table, crosstab
- FORBIDDEN: import, os, sys, eval, exec, open, read, write, __
- For date filtering: use pd.to_datetime() to parse date columns
- Always use EXACT column names from the provided list
- If the query is vague (e.g. "show data", "what's in here"), return df.head(10)
- If the query mentions "top N", use nlargest() or sort_values().head()
- If the query is about distribution or counts, use value_counts()
- If the query could mean multiple things, pick the most likely interpretation and execute it
- NEVER refuse to generate code. Always produce your best attempt."""

    prompt = f"""Dataframe Columns: {columns}
Column Data Types: {dtypes}
Sample Data (first 3 rows): {sample_data}
{semantic_str}
{history_str}

User Question: "{query}"

Think about what the user wants, then generate the Pandas code:"""

    response = call_llm(prompt, system_message=system_msg)

    # Extract code from markdown block
    match = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        code = response.strip()

    # Clean up common conversational prefixes
    code = code.replace("Here is the code:", "").replace("Here is the pandas code:", "").strip()
    # Remove any trailing explanation
    lines = code.split("\n")
    code_lines = [l for l in lines if not l.startswith("#") and l.strip()]
    if code_lines:
        code = "\n".join(code_lines)

    return code


def nl_to_pandas_with_retry(query, df, history=None, max_retries=2):
    """
    Generate Pandas code with automatic error correction retry loop. (Feature 2 enhancement)
    Returns: (code, result, error)
    """
    from visualize import execute_pandas_code_safely

    code = nl_to_pandas(query, df, history=history)
    result, executed_code, error = execute_pandas_code_safely(code, df)

    retries = 0
    while error and retries < max_retries:
        retries += 1
        retry_prompt = f"""The previous code failed with this error: {error}

Original query: "{query}"
Previous code that failed: {code}
Dataframe columns: {list(df.columns)}
Column types: {{col: str(df[col].dtype) for col in df.columns}}

Fix the code. Return ONLY the corrected Python code in a ```python block. Do not explain."""

        response = call_llm(retry_prompt, system_message="You are a Python code debugger. Fix the Pandas code. Return ONLY code in a ```python block.")
        
        match = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = response.strip()
        
        code = code.replace("Here is the code:", "").strip()
        result, executed_code, error = execute_pandas_code_safely(code, df)

    return code, result, error


# ──────────────────────────────────────────────
# AGENT 4: INSIGHT GENERATOR (Feature 4)
# ──────────────────────────────────────────────

INSIGHT_SYSTEM = """You are a data insight generator for non-technical business users.
Your job is to explain data results in simple, clear English.
Rules:
- NEVER mention pandas, dataframes, code, or technical terms
- Use specific numbers from the data
- Explain the "so what" — why does this matter?
- If showing changes, mention direction and magnitude
- Keep explanations concise (2-4 sentences)
- If the result is empty, say so clearly"""


def generate_explanation(query, df_used, final_result):
    """
    Generate short simple english description of what was computed.
    """
    res_str = str(final_result)
    if len(res_str) > 500:
        res_str = res_str[:500] + "...(truncated)"

    prompt = f"""What was computed: {query}
Result: {res_str}

Explain what this result means in 1-2 sentences. Focus on the key takeaway."""

    return call_llm(prompt, system_message=INSIGHT_SYSTEM)


def generate_answer_from_pandas(query, final_result):
    """
    Formulates a rich, conversational answer from the dataframe result.
    Returns a natural language summary with key figures highlighted.
    """
    if isinstance(final_result, pd.DataFrame):
        res_str = final_result.to_csv(index=False)
        row_count = len(final_result)
        col_count = len(final_result.columns)
        shape_hint = f"\n[Result shape: {row_count} rows × {col_count} columns]"
    elif isinstance(final_result, pd.Series):
        res_str = final_result.to_string()
        shape_hint = f"\n[Result: Series with {len(final_result)} entries]"
    else:
        res_str = str(final_result)
        shape_hint = f"\n[Result: single value]"

    if len(res_str) > 2500:
        res_str = res_str[:2500] + "\n...(truncated — summarize what is visible)"

    prompt = f"""User Question: "{query}"

Data Result:
{res_str}
{shape_hint}

Write a clear, helpful response following these rules:
1. Start with a DIRECT answer to the user's question using specific numbers from the data
2. Highlight the most important findings (top/bottom values, totals, patterns)
3. If there are interesting patterns or notable outliers, mention them briefly
4. If the data is empty or shows 0 rows, say "I couldn't find any matching data for that query."
5. Use bullet points or numbered lists for multiple data points
6. NEVER mention pandas, dataframes, code, or technical implementation details
7. Write as if you're a helpful data analyst explaining findings to a business user
8. Keep it concise but comprehensive — 2-5 sentences typically"""

    return call_llm(prompt, system_message=INSIGHT_SYSTEM)


def generate_insights(df):
    """
    Summarizes dataframe characteristics with meaningful insights.
    """
    stats = df.describe(include='all').to_string()
    if len(stats) > 2000:
         stats = stats[:2000] + "\n...[truncated]"

    prompt = f"""Dataset has {len(df)} rows and {len(df.columns)} columns.
Columns: {list(df.columns)}

Summary Statistics:
{stats}

Provide 3-5 interesting and actionable insights about this dataset. Focus on:
1. Notable patterns or distributions
2. Potential data quality issues
3. Key metrics and their ranges
4. Any surprising findings"""

    return call_llm(prompt, system_message=INSIGHT_SYSTEM)


# ──────────────────────────────────────────────
# AGENT 5: COMPARISON NARRATOR (Feature 9)
# ──────────────────────────────────────────────

def narrate_comparison(query, comparison_data):
    """
    Generate a human-readable narrative from comparison engine output.
    """
    prompt = f"""User asked: "{query}"

Comparison results:
- Period A ({comparison_data['period_a']['period']}): Total = {comparison_data['period_a']['sum']}, Avg = {comparison_data['period_a']['mean']}, Count = {comparison_data['period_a']['count']}
- Period B ({comparison_data['period_b']['period']}): Total = {comparison_data['period_b']['sum']}, Avg = {comparison_data['period_b']['mean']}, Count = {comparison_data['period_b']['count']}
- Change: {comparison_data['change']} ({comparison_data['change_pct']:+.1f}%)
- Direction: {comparison_data['direction']}

Write a clear 2-3 sentence comparison summary for a business user."""

    return call_llm(prompt, system_message=INSIGHT_SYSTEM)


# ──────────────────────────────────────────────
# AGENT 6: ROOT CAUSE NARRATOR (Feature 8)
# ──────────────────────────────────────────────

def narrate_root_cause(query, rca_data):
    """
    Generate a human-readable narrative from root cause analysis output.
    """
    if "error" in rca_data:
        return rca_data["error"]
    if "message" in rca_data:
        return rca_data["message"]

    top_drivers = rca_data.get("top_drivers", [])[:5]
    drivers_str = "\n".join([
        f"- {d['dimension']}/{d['segment']}: changed by {d['change']:+.2f} (contributing {d['contribution_pct']:+.1f}%)"
        for d in top_drivers
    ])

    prompt = f"""User asked: "{query}"

Overall Change: {rca_data['total_previous']} → {rca_data['total_current']} ({rca_data['total_change_pct']:+.1f}%)

Top Drivers:
{drivers_str}

Explain the root cause in 2-4 sentences. Focus on the top 2-3 drivers. Be specific."""

    return call_llm(prompt, system_message=INSIGHT_SYSTEM)


# ──────────────────────────────────────────────
# AGENT 7: BREAKDOWN NARRATOR (Feature 10)
# ──────────────────────────────────────────────

def narrate_breakdown(query, breakdown_data):
    """
    Generate a human-readable narrative from breakdown output.
    """
    if "error" in breakdown_data:
        return breakdown_data["error"]

    sections = []
    for dim, info in breakdown_data.get("breakdowns", {}).items():
        segs = info.get("segments", [])[:5]
        seg_str = ", ".join([f"{s[dim]}={s['share_pct']}%" for s in segs])
        sections.append(f"{dim}: {seg_str}")

    prompt = f"""User asked: "{query}"

Metric: {breakdown_data['metric']}, Grand Total: {breakdown_data['grand_total']}

Breakdown:
{chr(10).join(sections)}

Summarize the breakdown in 2-3 sentences. Highlight the dominant contributors and any outliers."""

    return call_llm(prompt, system_message=INSIGHT_SYSTEM)


# ──────────────────────────────────────────────
# AGENT 8: AUTO SUMMARY NARRATOR (Feature 11)
# ──────────────────────────────────────────────

def narrate_summary(query, summary_data):
    """
    Generate a comprehensive data summary narrative.
    """
    prompt = f"""User asked: "{query}"

Dataset Overview:
- Shape: {summary_data['shape']['rows']} rows × {summary_data['shape']['columns']} columns
- Columns: {summary_data['columns']}

Numeric Summary: {json.dumps(summary_data.get('numeric_summary', {}), indent=2)[:800]}

Categorical Summary: {json.dumps(summary_data.get('categorical_summary', {}), indent=2)[:500]}

Date Range: {json.dumps(summary_data.get('date_range', {}))}

Notable Patterns: {summary_data.get('notable_patterns', [])}

Missing Data: {json.dumps(summary_data.get('missing_data', {}))[:300]}

Generate a concise executive summary (4-6 sentences) covering key metrics, trends, and any concerns."""

    return call_llm(prompt, system_message=INSIGHT_SYSTEM)


# ──────────────────────────────────────────────
# AGENT 9: ANOMALY NARRATOR (Feature 17)
# ──────────────────────────────────────────────

def narrate_anomalies(query, anomaly_data):
    """
    Generate a human-readable narrative from anomaly detection output.
    """
    if not anomaly_data.get("has_anomalies"):
        return "No significant anomalies were detected in the data. All values fall within expected ranges."

    anomaly_details = []
    for col, info in anomaly_data.get("anomalies", {}).items():
        anomaly_details.append(
            f"- {col}: {info['count']} anomalies ({info['percent']}% of data). "
            f"Bounds: [{info['lower_bound']}, {info['upper_bound']}]. "
            f"High outliers: {info['direction']['high']}, Low outliers: {info['direction']['low']}"
        )

    prompt = f"""User asked: "{query}"

Anomaly Detection Results ({anomaly_data['method'].upper()} method):
Total rows analyzed: {anomaly_data['total_rows']}

{chr(10).join(anomaly_details)}

Summarize the anomalies found in 2-4 sentences. Explain which columns have unusual patterns and what they might indicate."""

    return call_llm(prompt, system_message=INSIGHT_SYSTEM)


# ──────────────────────────────────────────────
# SMART METRIC/COLUMN DETECTION
# ──────────────────────────────────────────────

def detect_metric_and_dimensions(query, df):
    """
    Use LLM to identify which column is the metric and which are dimensions
    for analysis queries (comparison, root cause, breakdown).
    Returns: (metric_col, dimension_cols, period_col)
    """
    columns = list(df.columns)
    dtypes = {col: str(df[col].dtype) for col in df.columns}

    prompt = f"""Given this query: "{query}"
And these columns with types: {json.dumps(dtypes)}

Identify:
1. metric_col: The main numeric column being analyzed (e.g., revenue, sales, profit)
2. period_col: The time/date column if comparing periods (or null)
3. dimension_cols: Categorical columns to break down by (list)

Reply in this exact JSON format:
{{"metric_col": "column_name", "period_col": "column_name_or_null", "dimension_cols": ["col1", "col2"]}}"""

    response = call_llm(prompt, temperature=0.0)

    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            metric = parsed.get("metric_col")
            period = parsed.get("period_col")
            dims = parsed.get("dimension_cols", [])

            # Validate columns exist
            if metric and metric not in columns:
                # Fuzzy match
                for c in columns:
                    if metric.lower() in c.lower() or c.lower() in metric.lower():
                        metric = c
                        break
                else:
                    # Default to first numeric column
                    numeric = df.select_dtypes(include="number").columns
                    metric = numeric[0] if len(numeric) > 0 else None

            if period == "null" or period == "None":
                period = None
            if period and period not in columns:
                period = None

            dims = [d for d in dims if d in columns and d != metric and d != period]

            return metric, dims, period
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: first numeric col as metric
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in columns if c not in numeric][:3]
    return (numeric[0] if numeric else None, categorical, None)


# ──────────────────────────────────────────────
# VISION ANALYSIS (preserved from original)
# ──────────────────────────────────────────────

def analyze_vision_chart(query, image_bytes):
    """Analyze an uploaded chart/image using Llama 3.2 Vision."""
    try:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-90b-vision-preview",
            temperature=0.2,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Vision analysis error: {e}"
