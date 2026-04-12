"""
export_engine.py — Export & Sharing Module for Talk-to-Data
Handles: PDF report generation (via reportlab), CSV export.
"""

import io
import os
import tempfile
from datetime import datetime

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable
    )
    from reportlab.graphics.shapes import Drawing, Line
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def export_dataframe_to_csv(df):
    """Export a DataFrame as CSV bytes for download."""
    if df is None:
        return None
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _save_chart_to_image(fig):
    """Save a matplotlib figure to a temporary PNG file and return the path."""
    if fig is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    tmp.close()
    return tmp.name


def export_to_pdf(title, query, response_text, chart_fig=None, dataframe=None, trust_info=None, chat_history=None):
    """
    Generate a PDF report with the chat history or single query/response, chart, and data table.
    Returns PDF bytes.
    """
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=50, leftMargin=50,
        topMargin=60, bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#6C63FF"),
        spaceAfter=20,
        fontName="Helvetica-Bold",
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#E94560"),
        spaceBefore=15,
        spaceAfter=8,
        fontName="Helvetica-Bold",
    )
    user_style = ParagraphStyle(
        "UserStyle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#333333"),
        spaceAfter=10,
        leading=16,
        fontName="Helvetica-Bold",
    )
    ai_style = ParagraphStyle(
        "AIStyle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#333333"),
        spaceAfter=10,
        leading=16,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#888888"),
        spaceAfter=5,
    )

    elements = []

    # Title
    elements.append(Paragraph(title or "Talk-to-Data Report", title_style))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        meta_style
    ))
    elements.append(HRFlowable(
        width="100%", thickness=2,
        color=colors.HexColor("#6C63FF"), spaceAfter=20
    ))

    # Conversation History
    elements.append(Paragraph("Conversation History", heading_style))
    
    tmp_files_to_clean = []

    if chat_history and len(chat_history) > 0:
        for msg in chat_history:
            role_label = "User:" if msg.get("role") == "user" else "AI:"
            style = user_style if msg.get("role") == "user" else ai_style
            text = msg.get("content", "")
            
            # For AI responses, they might have multiple paragraphs
            if msg.get("role") == "user":
                elements.append(Paragraph(f'<b>{role_label}</b> "{text}"', style))
            else:
                elements.append(Paragraph(f"<b>{role_label}</b>", user_style))
                for para in text.split("\n"):
                    if para.strip():
                        elements.append(Paragraph(para.strip(), style))
                
                # Render chart_data if present
                chart_data = msg.get("chart_data")
                if chart_data and isinstance(chart_data, list) and len(chart_data) > 0:
                    try:
                        import matplotlib.pyplot as plt
                        from visualize import _apply_dark_theme, COLORS
                        df_c = pd.DataFrame(chart_data)
                        if not df_c.empty:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ctype = msg.get("chart_type", "").lower()
                            title = msg.get("chart_keys", ["Value"])[0] if msg.get("chart_keys") else "Chart"
                            
                            cols = df_c.columns
                            x_col = cols[0]
                            
                            if "anomaly" in ctype or ctype == "scatter":
                                # Anomaly or scatter plot
                                x_vals = df_c["index"] if "index" in cols else df_c[cols[0]]
                                y_vals = df_c["value"] if "value" in cols else df_c[cols[1]]
                                ax.scatter(x_vals, y_vals, color=COLORS[1], alpha=0.6)
                                if "is_anomaly" in cols:
                                    anomalies = df_c[df_c["is_anomaly"] == True]
                                    if not anomalies.empty:
                                        a_x = anomalies["index"] if "index" in anomalies else anomalies[cols[0]]
                                        a_y = anomalies["value"] if "value" in anomalies else anomalies[cols[1]]
                                        ax.scatter(a_x, a_y, color="#E94560", label="Anomaly", zorder=5)
                                        ax.legend(facecolor="#1e1e2d", edgecolor="#333", labelcolor="white")
                            elif ctype == "pie":
                                y_col = "value" if "value" in cols else cols[1]
                                ax.pie(df_c[y_col], labels=df_c[x_col], autopct="%1.1f%%", colors=COLORS)
                            elif ctype == "line":
                                y_col = "value" if "value" in cols else cols[1]
                                ax.plot(df_c[x_col].astype(str).head(15), df_c[y_col].head(15), marker='o', color=COLORS[2])
                            else:
                                # Default bar
                                y_col = "value" if "value" in cols else cols[1]
                                ax.bar(df_c[x_col].astype(str).head(15), df_c[y_col].head(15), color=COLORS[0])
                            
                            ax.set_title(title.title(), color="white", fontsize=11)
                            _apply_dark_theme(fig, ax)
                            plt.tight_layout(pad=2.0)
                            
                            c_path = _save_chart_to_image(fig)
                            plt.close(fig)
                            
                            if c_path:
                                tmp_files_to_clean.append(c_path)
                                elements.append(Spacer(1, 10))
                                try:
                                    elements.append(RLImage(c_path, width=4.5 * inch, height=3.0 * inch))
                                except Exception:
                                    pass
                    except Exception as e:
                        print("Failed to plot chart inline:", e)
                        
            elements.append(Spacer(1, 10))
    else:
        # Fallback to single interaction
        if query:
            elements.append(Paragraph(f'<b>User:</b> "{query}"', user_style))
            elements.append(Spacer(1, 5))

        if response_text:
            elements.append(Paragraph("<b>AI:</b>", user_style))
            for para in response_text.split("\n"):
                if para.strip():
                    elements.append(Paragraph(para.strip(), ai_style))
            elements.append(Spacer(1, 5))

    # Chart
    if chart_fig:
        chart_path = _save_chart_to_image(chart_fig)
        if chart_path:
            tmp_files_to_clean.append(chart_path)
            elements.append(Spacer(1, 15))
            elements.append(Paragraph("Visualization", heading_style))
            try:
                img = RLImage(chart_path, width=5.5 * inch, height=3.5 * inch)
                elements.append(img)
            except Exception:
                pass

    # Data Table (limited rows)
    if dataframe is not None and isinstance(dataframe, (pd.DataFrame, pd.Series)):
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Data Preview", heading_style))

        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.reset_index()
            dataframe.columns = ["Index", "Value"]

        # Limit to 20 rows for PDF
        display_df = dataframe.head(20)
        table_data = [list(display_df.columns)]
        for _, row in display_df.iterrows():
            table_data.append([str(v)[:40] for v in row.values])  # Truncate long values

        col_count = len(display_df.columns)
        col_width = min(6.5 * inch / max(col_count, 1), 2 * inch)

        t = Table(table_data, colWidths=[col_width] * col_count)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C63FF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5FF")]),
        ]))
        elements.append(t)

        if len(dataframe) > 20:
            elements.append(Paragraph(
                f"<i>Showing 20 of {len(dataframe)} rows</i>", meta_style
            ))

    # Trust Layer Info
    if trust_info:
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Transparency Details", heading_style))
        if trust_info.get("pandas_code"):
            elements.append(Paragraph("Generated Code:", ai_style))
            code_style = ParagraphStyle(
                "Code", parent=styles["Code"],
                fontSize=8, backColor=colors.HexColor("#F0F0F0"),
                leftIndent=10, rightIndent=10,
            )
            elements.append(Paragraph(trust_info["pandas_code"], code_style))
        if trust_info.get("explanation"):
            elements.append(Paragraph(f"Explanation: {trust_info['explanation']}", ai_style))

    # Footer
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor("#CCCCCC"), spaceAfter=10
    ))
    elements.append(Paragraph(
        "Generated by Talk-to-Data AI Assistant",
        ParagraphStyle("Footer", parent=meta_style, alignment=TA_CENTER)
    ))

    # Build PDF
    try:
        doc.build(elements)
    finally:
        # Clean up temp files AFTER build
        for path in tmp_files_to_clean:
            try:
                os.unlink(path)
            except Exception:
                pass

    buffer.seek(0)
    return buffer.getvalue()
