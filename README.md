# 🌟 Talk-to-Data AI Assistant

## i. Overview
**Talk-to-Data Assistant** (formerly MultiRAG) is a multimodal Retrieval-Augmented Generation application that intelligently answers questions across distinct data formats. It solves the massive problem of disparate corporate data silos by seamlessly handling both structured data (like CSV spreadsheets) and unstructured data (like audio, text, PDFs, and images) simultaneously within an accessible ChatGPT-like interface. The intended users are data analysts, researchers, and enterprise workers who need rapid, explainable insights from vast knowledge sources without writing custom SQL or complex Python pipelines.

## ii. Features
- **Conversational Chat Interface:** A fluid, conversational UI allowing direct QA querying onto uploaded data.
- **Natural Language to Pandas Engine:** Automatically converts natural language questions directly to secure Pandas computation strings evaluated natively in Python.
- **Intelligent Query Router:** Automatically disambiguates if a user's prompt is meant to target tabular structured spreadsheet data, or unstructured text/resume embeddings.
- **Strict Semantic Dictionary:** Injects dataset-specific metric definitions into the execution prompt, ensuring that ambiguous terms like "Active Users" or "Net Revenue" are calculated with mathematical consistency every single time, preventing LLM hallucination in data operations.
- **Trust Layer and Explainability:** Displays the generated Python code, verbatim RAG context chunks fetched, and visual dataframe previews to ensure full transparency on the AI's logic paths.
- **Pre-emptive Auto-Charting:** Understands charting intents (e.g. "plot the distribution") and automatically renders native `matplotlib` graphs representing the isolated dataframe.
- **Chart & Dashboard Analyzer (Vision LLM):** Use the chat's native paperclip feature to upload static BI dashboard screenshots or charts. The AI bypasses structured frameworks via its Llama-3.2 Vision node to natively "read" graphs directly, proving immense multi-modal utility for non-technical workers lacking raw source logic.
- **Multimodal Upload Logic:** Imports CSVs, Websites natively through scrapers, PDFs, Audio files via Whisper transcription, and visual queries via OpenAI CLIP.

## iii. Install and Run Instructions
*Note: Make sure Python 3.11+ is installed on your OS.*

1. **Clone the repository:**
   ```bash
   git clone https://github.com/safalsingh1/MultiRAG.git
   cd MultiRAG
   ```
2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   # source venv/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables Check:**
   Copy the `.env.example` file to create a live `.env` file, and insert your actual `GROQ_API_KEY`.
   ```bash
   cp .env.example .env
   ```
5. **Run the application:**
   ```bash
   python -m streamlit run app.py
   ```
   *The application will automatically pop up in your default web browser at `http://localhost:8501`.*

## iv. Tech Stack
* **Programming Languages**: Python
* **Frameworks**: Streamlit (Frontend GUI), Pandas / Matplotlib (Data processing & visualization)
* **Databases**: FAISS (Local Vector Database)
* **Cloud Services/APIs**: Groq Cloud inference endpoints
* **AI/ML Libraries and Models**:
  * `llama-3.3-70b-versatile` *(Via Groq - Used as fundamental Intent Routing and Logic Generator)*
  * `OpenAI CLIP` *(Image embedding models)*
  * `OpenAI Whisper` *(Audio transcription models)*
  * `SentenceTransformers` (`all-MiniLM-L6-v2` for raw text indexing)

## v. Usage Examples
Once the dashboard opens in your browser on `localhost:8501`, follow these examples:

* **Example 1: Analyzing CSV structured Tables**
  * Use the Left Sidebar to upload any CSV file (e.g. `sales_records.csv`).
  * In the main Chat Input, type: *"What is the sum of total profits grouped by item type? Plot it as a chart."*
  * The system translates this directly into a `.groupby()` Pandas execution, cleanly executes it safely, formats the output visually as a chart, and gives you a conversational answer on the result.
* **Example 2: Analyzing text or image Unstructured RAG**
  * Use the Left Sidebar's "Unstructured Data" dropdown to select "Website Link". Enter a Wikipedia URL.
  * In the main Chat input, type: *"Summarize the early life of the person in the article."*
  * The smart router dynamically averts the Pandas engine to target the FAISS vector database natively and fetches context to answer you perfectly.
* **Example 3: Visual Chart & Dashboard Analyzer**
  * Click the paperclip icon in the main Chat Input to attach any static screenshot of a legacy BI dashboard, or revenue graph.
  * In the chat, ask: *"What trend do you see causing this massive sudden spike in Q2?"*
  * The Llama-3.2 Multimodal Vision backend inherently decodes the visual nodes, colors, and layout elements of the raw image directly into a deep conversational explanation!

## vi. Optional Details

### Architecture Notes
Our project relies on a deeply componentized structure ensuring isolation between the execution algorithms and user interfaces:
* `app.py`: Acts as the frontend application framework handling states and routing visual blocks (e.g., Trust Layers, sidebars).
* `llm_engine.py`: Defines abstract translation matrices that translate local states into prompts for the Groq cloud. It hosts the Smart Intent Router, the Semantic Dictionary engine, and the hallucination-prevention NLP translators.
* `visualize.py`: Strictly houses the isolated dynamic execution sandboxes blocking `eval()` attacks, restricting system access entirely while retrieving local dictionaries.
* `vectordb.py` / `utils.py`: Contains configurations for setting up and querying the local FAISS indices.

### Context Window Scalability
To handle massive datasets without exploding context windows, the CSV schema passed to Groq strictly bounds payloads. We map dataset shapes inherently via `columns` constraints and pass exactly `.head(3)` as representative JSON samples. The LLM never absorbs the entire database; it builds operations precisely around the metadata boundaries instead.

### Limitations
* Extreme multi-line spreadsheet calculations aren't executed. Because of our strict sandbox security guidelines within `visualize.py`, the AI engine is forcefully restricted to executing extremely safe one-liner expressions like `.query()` or `.sort_values()` natively avoiding code-execution jailbreaks.
* There are hard ceilings built into the contextual mapping size when attempting to parse massive CSV configurations over simple prompt calls. String mappings deliberately truncate columns to avoid maxing out active parameter lengths across models.

### Future Improvements
* Shifting natively from FAISS entirely toward cloud-serviced vector endpoints (i.e. Pinecone) to drastically improve concurrent horizontal query speeds.
* Building out a more robust SQL-centric adapter directly for massive scale data stores natively beyond CSV.
