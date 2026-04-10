import pandas as pd
import re
import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

MODEL = "llama-3.3-70b-versatile"

def call_llm(prompt, model=MODEL):
    """
    Calls the Groq API.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.0
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"Error connecting to LLM: {e}"

def route_intent(query, has_df=False, has_rag=False):
    """
    Classifies intent of query between 'structured' (CSV/pandas) and 'rag' (unstructured).
    """
    query_lower = query.lower()
    
    structured_keywords = ["average", "sum", "count", "mean", "median", "plot", "chart", "trend", "csv", "data", "table", "dataframe", "columns", "sort", "filter", "group by", "histogram", "bar", "line"]
    rag_keywords = ["document", "pdf", "image", "audio", "website", "text", "chunk", "context"]
    
    struct_score = sum(1 for k in structured_keywords if k in query_lower)
    rag_score = sum(1 for k in rag_keywords if k in query_lower)
    
    if has_df and not has_rag:
        return "structured"
    if has_rag and not has_df:
        return "rag"
    
    if struct_score > rag_score:
        return "structured"
    elif rag_score > struct_score:
        return "rag"
        
    # If ambiguous and both sources exist, ask the LLM to route it
    if has_df and has_rag:
        prompt = f"Does the following query sound like it's asking for calculations on a spreadsheet/CSV (reply 'structured') or asking about general documents/resumes/text (reply 'rag')? Query: {query}"
        ans = call_llm(prompt).lower()
        if 'rag' in ans:
            return 'rag'
        return 'structured'
        
    return "structured" if has_df else "rag"

import json
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
    Translates Natural Language to Pandas code using Gemma.
    Allowed operations logically restrict output code.
    Returns python code string.
    """
    columns = list(df.columns)
    sample_data = df.head(3).to_dict()
    
    history_str = ""
    if history:
        history_str = "Chat History:\n" + "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
        
    semantic_dict = load_semantic_dict()
    semantic_str = ""
    if semantic_dict:
        semantic_str = "\nStrict Business Metric Definitions (USE THESE EXACT MATHEMATICAL RULES IF ASKED):\n" + json.dumps(semantic_dict, indent=2) + "\n"
    
    prompt = f"""You are a data analyst assistant. Convert the user's natural language query into exactly ONE line of Python Pandas code. 
The dataframe is named `df`. 
The target logic must produce a pandas Series, DataFrame, or scalar. 
Allowed Pandas operations: groupby, sum, mean, count, sort_values, sort_index, filtering (e.g., df[df['col'] > 0]), and basic math.
DO NOT use imports, os, sys, eval, exec, or write complex loops. 
Return ONLY the python code inside a ```python block, nothing else.

Dataframe Columns: {columns}
Sample Data: {sample_data}
{semantic_str}
{history_str}
Query: {query}
Python Code:"""

    response = call_llm(prompt)
    
    # Extract code from markdown block
    match = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        # Fallback if no code block
        code = response.strip()
        
    # Clean up common conversational prefixes
    code = code.replace("Here is the code:", "").replace("Here is the pandas code:", "").strip()
    
    return code

def generate_explanation(query, df_used, final_result):
    """
    Generate short simple english description of what was computed.
    """
    res_str = str(final_result)
    if len(res_str) > 500:
        res_str = res_str[:500] + "...(truncated)"
        
    prompt = f"""Explain this pandas logic calculation briefly in 1 sentence. Focus on mechanics.
What was the question: {query}
Result snippet: {res_str}

Explanation:"""
    return call_llm(prompt)

def generate_answer_from_pandas(query, final_result):
    """
    Formulates a conversational answer reading the dataframe result.
    """
    if isinstance(final_result, pd.DataFrame):
        # Use csv format to natively bypass pandas default column truncation without requiring 'tabulate'
        res_str = final_result.to_csv(index=False)
    elif isinstance(final_result, pd.Series):
        res_str = final_result.to_string()
    else:
        res_str = str(final_result)
        
    if len(res_str) > 2500:
        res_str = res_str[:2500] + "\n...(truncated due to size. summarize what is available)"
        
    prompt = f"""You are a helpful data assistant. The user asked a question, and the backend calculated this data result.
If the Data Result is empty or shows 0 rows, explicitly say "I couldn't find any data matching that request in the table." DO NOT guess or hallucinate details.
Otherwise, answer the user's question directly and comprehensively using ONLY information from the data result. Do not mention pandas or dataframes. Formulate a rich, natural sounding paragraph.

User Question: {query}
Data Result: 
{res_str}

Conversational Answer:"""
    return call_llm(prompt)

def generate_insights(df):
    """
    Summarizes dataframe characteristics.
    """
    stats = df.describe(include='all').to_string()
    if len(stats) > 2000:
         stats = stats[:2000] + "\n...[truncated]"
         
    prompt = f"""You are a data analyst. I am providing you with the summary statistics of a dataset.
Describe 3 interesting insights, trends, or anomalies about this dataset in simple English. Keep it concise.

Summary Stats:
{stats}

Insights:"""
    return call_llm(prompt)
