import matplotlib.pyplot as plt
import pandas as pd
import io
import re

def execute_pandas_code_safely(code: str, df: pd.DataFrame):
    """
    Safely executes generated pandas code.
    Blocks any string with imports, os, sys, direct evals, dunders, etc.
    """
    # 1. Reject fundamentally unsafe patterns
    unsafe_patterns = ["import", "os", "sys", "eval", "exec", "__", "open", "read", "write"]
    for pattern in unsafe_patterns:
        if pattern in code:
            raise ValueError(f"Security Alert: Unsafe operation detected ({pattern}). Blocked execution.")
            
    # 2. Strict evaluation dict
    local_vars = {}
    
    # 3. Execute in sandbox explicitly mapping pd and df
    try:
        # Try evaluating as a pure expression first (e.g. df.sort_values(by='Customer Id', ascending=False))
        result = eval(code, {"pd": pd, "df": df}, local_vars)
        return result, code, None
    except SyntaxError:
        # If it's a statement (like an assignment rule), execute it
        try:
            exec(code, {"pd": pd, "df": df}, local_vars)
            
            # Retrieve the most recently assigned custom variable
            assigned_vars = [v for k, v in local_vars.items() if not k.startswith('_')]
            if assigned_vars:
                return assigned_vars[-1], code, None
            
            # If no variable was assigned, assume an inplace modification happened and return df
            return df, code, None
        except Exception as e:
            return None, code, f"Execution error: {str(e)}"
    except Exception as e:
        return None, code, f"Evaluation error: {str(e)}"

def generate_auto_chart(result_data):
    """
    Takes a pandas Result (Series or DataFrame) and automatically graphs it.
    Returns a matplotlib Figure or None if unplottable.
    """
    if result_data is None:
        return None
        
    fig, ax = plt.subplots(figsize=(8, 5))
    
    try:
        if isinstance(result_data, pd.Series):
            if len(result_data) < 20:
                result_data.plot(kind='bar', ax=ax, title="Data Bar Chart")
            else:
                result_data.plot(kind='line', ax=ax, title="Data Trend")
        elif isinstance(result_data, pd.DataFrame):
            # Check row count and column count
            numeric_cols = result_data.select_dtypes(include='number').columns
            
            if len(numeric_cols) > 0:
                if len(result_data) < 20:
                     # Bar chart
                     result_data[numeric_cols].plot(kind='bar', ax=ax, title="Comparison Chart")
                else:
                     # Trend Chart
                     result_data[numeric_cols].plot(kind='line', ax=ax, title="Trend Line Chart")
            else:
                return None
        else:
            # It's a scalar or unplottable
            return None
            
        plt.tight_layout()
        return fig
    except Exception as e:
         print(f"Graphing failed: {e}")
         return None
         
def get_auto_explanation(result_data):
     """ Generates an automatic simple english string describing the table state natively without LLM."""
     if isinstance(result_data, pd.Series):
          return f"Computed a series with {len(result_data)} rows. Index: {result_data.index.name or 'values'}."
     elif isinstance(result_data, pd.DataFrame):
          return f"Computed a Dataframe with {len(result_data)} rows and {len(result_data.columns)} columns."
     else:
          return f"Calculated value: {result_data}"
