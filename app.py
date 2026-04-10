import os
import pandas as pd
import streamlit as st
import torch
from pathlib import Path

from data_upload.input_sources_utils import image_util, pdf_util, website_util, audio_util
from utils import load_clip_model, load_text_embedding_model, load_whisper_model, load_text_index, load_audio_index, load_image_index
from vectordb import search_text_index, search_image_index

from llm_engine import call_llm, route_intent, nl_to_pandas, generate_explanation, generate_insights, generate_answer_from_pandas
from visualize import execute_pandas_code_safely, generate_auto_chart, get_auto_explanation

os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(layout="wide", page_title="Talk-to-Data Assistant", page_icon="🤖")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_clip_model()
text_embedding_model = load_text_embedding_model()
whisper_model = load_whisper_model()

# Handle missing directories
_db_dirs = ["./vectorstore", "./images"]
for d in _db_dirs:
    Path(d).mkdir(parents=True, exist_ok=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
if "df" not in st.session_state:
    st.session_state["df"] = None

with st.sidebar:
    st.title("Talk-to-Data Assistant")
    st.write("Upload your data and chat directly with it!")
    
    st.header("1. Structured Data (CSV)")
    uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"], key="csv_uploader")
    if uploaded_csv is not None:
        try:
            st.session_state["df"] = pd.read_csv(uploaded_csv)
            st.success(f"CSV Loaded: {len(st.session_state['df'])} rows.")
            
            if st.button("Generate Generic Insights"):
                with st.spinner("Generating Insights..."):
                    insights = generate_insights(st.session_state["df"])
                    st.session_state.messages.append({"role": "assistant", "content": insights})
        except Exception as e:
            st.error(f"Error loading CSV: {e}")

    st.divider()
    
    st.header("2. Unstructured Data (RAG)")
    upload_choice = st.selectbox(
        options=["Upload PDF", "Website Link", "Upload Image", "Add Image from URL", "Audio Recording"], 
        label="Select Upload Type"
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


# --- MAIN CHAT INTERFACE ---
st.title("Data Chat 💬")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "uploaded_images" in msg and msg["uploaded_images"]:
            for img_bytes in msg["uploaded_images"]:
                st.image(img_bytes, width=400)
        
        # Display Trust Layer artifacts if available
        if "trust_layer" in msg:
            with st.expander("🔍 Trust Layer: Explainability & Insights"):
                tl = msg["trust_layer"]
                if tl.get("pandas_code"):
                     st.subheader("Generated Pandas Code")
                     st.code(tl["pandas_code"], language="python")
                if tl.get("dataframe_preview") is not None:
                     st.subheader("Data Result Preview")
                     try:
                         st.dataframe(tl["dataframe_preview"])
                     except Exception:
                         # Fallback if the user's dataset has mixed/corrupted datatypes breaking PyArrow 
                         st.dataframe(tl["dataframe_preview"].astype(str))
                if tl.get("explanation"):
                     st.subheader("Compute Explanation")
                     st.write(tl["explanation"])
                if tl.get("sources"):
                     st.subheader("RAG Source Chunks")
                     for idx, c in enumerate(tl["sources"]):
                           st.write(f"**Chunk {idx+1}:** {c}")
                           st.divider()
        if "chart" in msg and msg["chart"] is not None:
            st.pyplot(msg["chart"])

st.sidebar.divider()
st.sidebar.header("3. Query Settings")
routing_mode = st.sidebar.radio("Routing Mode", ["Auto-Detect", "Structured (CSV)", "Unstructured (RAG)"], index=0, help="Force the AI to search only your CSV or only your documents.")

st.sidebar.divider()
st.sidebar.header("4. Semantic Dictionary")
import json
dict_path = "semantic_dict.json"
metrics_dict = {}
if os.path.exists(dict_path):
    try:
        with open(dict_path, "r") as f:
            metrics_dict = json.load(f)
    except:
        pass

with st.sidebar.expander("View / Edit Business Metrics", expanded=False):
    st.caption("Ensure the Artificial Intelligence runs mathematically consistent business formulations by defining your metrics below:")
    if metrics_dict:
        for k, v in metrics_dict.items():
            st.markdown(f"**{k}**: {v}")
    else:
        st.write("No custom metrics defined.")
        
    st.divider()
    new_metric_name = st.text_input("Metric Name (e.g., Gross Churn)")
    new_metric_def = st.text_area("Mathematical Definition", height=68)
    if st.button("Add Metric", use_container_width=True):
        if new_metric_name and new_metric_def:
            metrics_dict[new_metric_name] = new_metric_def
            with open(dict_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)
            st.success(f"Added {new_metric_name}!")
            st.rerun()

query_prompt = st.chat_input("Ask a question about your data...", accept_file="multiple", file_type=["png", "jpg", "jpeg"])

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
         
    # check capabilities
    has_df = st.session_state["df"] is not None
    has_rag_text = os.path.exists("./vectorstore/text_index.index")
    has_rag_audio = os.path.exists("./vectorstore/audio_index.index")
    has_rag_image = os.path.exists("./vectorstore/image_index.index")
    has_rag = has_rag_text or has_rag_audio or has_rag_image
    
    if not has_df and not has_rag:
        with st.chat_message("assistant"):
            warning_msg = "Please upload some data (CSV or Unstructured) from the sidebar first."
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
        st.stop()
        
    if files:
        intent = "vision"
    elif routing_mode == "Structured (CSV)":
        intent = "structured"
    elif routing_mode == "Unstructured (RAG)":
        intent = "rag"
    else:
        intent = route_intent(query, has_df, has_rag)
    
    with st.chat_message("assistant"):
         with st.spinner(f"Processing query as [{intent.upper()}] intent..."):
             response_content = ""
             trust_layer = {}
             chart_fig = None
             retrieved_images = []
             
             if intent == "vision":
                 from llm_engine import analyze_vision_chart
                 res = analyze_vision_chart(query, files[0].getvalue())
                 response_content = res
                 trust_layer["explanation"] = "Chart analyzed directly through natively injected multimodal capabilities via Llama 3.2 Vision."
             
             elif intent == "structured":
                 if not has_df:
                     response_content = "You asked a structured query, but no CSV is uploaded. Please upload a CSV."
                 else:
                     history = st.session_state.messages[-5:] if len(st.session_state.messages) >=5 else st.session_state.messages
                     code = nl_to_pandas(query, st.session_state["df"], history=history)
                     trust_layer["pandas_code"] = code
                     
                     df_result, code_run, err = execute_pandas_code_safely(code, st.session_state["df"])
                     if err:
                         response_content = f"Error executing query: {err}"
                     else:
                         # Attempt chart only if desired
                         wants_chart = any(k in query.lower() for k in ["chart", "plot", "graph", "trend", "visualize", "distribution", "histogram", "bar", "line"])
                         chart_fig = generate_auto_chart(df_result) if wants_chart else None
                         
                         explanation = generate_explanation(query, None, df_result)
                         trust_layer["explanation"] = explanation
                         
                         if isinstance(df_result, pd.DataFrame) or isinstance(df_result, pd.Series):
                             trust_layer["dataframe_preview"] = df_result
                             # Generate a dynamic conversational answer based on the returned data
                             response_content = generate_answer_from_pandas(query, df_result)
                         else:
                             response_content = generate_answer_from_pandas(query, df_result)
             else:
                 # RAG Intent
                 context_chunks = []
                 
                 if has_rag_text:
                     try:
                         ti, td = load_text_index()
                         res_idxs = search_text_index(query, ti, text_embedding_model, k=3)
                         for idx in res_idxs[0]:
                             if 0 <= idx < len(td):
                                 context_chunks.append(td['content'].iloc[idx])
                     except Exception as e:
                         pass
                 if has_rag_audio:
                     try:
                         ai, ad = load_audio_index()
                         res_idxs = search_text_index(query, ai, text_embedding_model, k=2)
                         for idx in res_idxs[0]:
                             if 0 <= idx < len(ad):
                                 context_chunks.append(ad['content'].iloc[idx])
                     except Exception as e:
                         pass
                         
                 # Only search for images if the user is explicitly looking for visual context
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
                     except Exception as e:
                         pass
                         
                 if not context_chunks and not retrieved_images:
                     response_content = "Could not find any relevant text or image context for your question in the uploaded documents."
                 else:
                     combined_context = " ".join(context_chunks)[:1500]
                     prompt = f"Answer the question based on the context below.\n\nContext: {combined_context}\n\nQuestion: {query}\n\nAnswer:"
                     
                     if "Image found" in combined_context and len(context_chunks) == 1:
                         ans = "I found this image matching your specific request!"
                     else:
                         ans = call_llm(prompt)
                     response_content = ans
                     trust_layer["sources"] = context_chunks
                     trust_layer["explanation"] = "Information and images retrieved from vector database via multimodal semantic search."

             # Display the final output
             st.markdown(response_content)
             
             if trust_layer:
                  with st.expander("🔍 Trust Layer: Explainability & Insights"):
                       if trust_layer.get("pandas_code"):
                            st.subheader("Generated Pandas Code")
                            st.code(trust_layer["pandas_code"], language="python")
                       if trust_layer.get("dataframe_preview") is not None:
                            st.subheader("Data Result Preview")
                            st.dataframe(trust_layer["dataframe_preview"])
                       if trust_layer.get("explanation"):
                            st.subheader("Compute Explanation")
                            st.write(trust_layer["explanation"])
                       if trust_layer.get("sources"):
                            st.subheader("RAG Source Chunks")
                            for idx, c in enumerate(trust_layer["sources"]):
                                  st.write(f"**Chunk {idx+1}:** {c}")
                                  st.divider()
             
             if chart_fig:
                  st.pyplot(chart_fig)
                  
             if retrieved_images:
                  for img_path in retrieved_images:
                      try:
                          st.image(img_path, caption="Retrieved relevant image")
                      except:
                          pass
                  
             # Append to history
             st.session_state.messages.append({
                 "role": "assistant", 
                 "content": response_content,
                 "trust_layer": trust_layer,
                 "chart": chart_fig,
                 "retrieved_images": retrieved_images if 'retrieved_images' in locals() else []
             })
