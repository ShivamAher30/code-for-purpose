"""
excel_util.py — Excel file upload handler for Talk-to-Data
Supports .xlsx and .xls files with multi-sheet selection.
"""

import pandas as pd
import streamlit as st


def upload_excel():
    """
    Streamlit widget for Excel file upload.
    Handles multi-sheet workbooks by letting the user pick a sheet.
    Returns the selected DataFrame or None.
    """
    uploaded_file = st.file_uploader(
        "Upload an Excel file", type=["xlsx", "xls"], key="excel_uploader"
    )

    if uploaded_file is not None:
        try:
            # Read all sheet names
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names

            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "Select a sheet", sheet_names, key="excel_sheet_select"
                )
            else:
                selected_sheet = sheet_names[0]

            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            st.success(f"Excel Loaded: {len(df)} rows from sheet '{selected_sheet}'.")
            return df

        except Exception as e:
            st.error(f"Error loading Excel file: {e}")
            return None

    return None
