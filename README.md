# Talk-to-Data

A conversational analytics platform that enables natural language interaction with structured and unstructured data sources. The system translates plain-English queries into executable Pandas operations, performs automated statistical analysis, and renders interactive visualizations — all through a chat-based interface backed by a multi-agent LLM architecture.

---

## Prerequisites

- **Python** 3.10 or 3.11 (required for `faiss-cpu` compatibility)
- **Node.js** 18+ and npm
- **Groq API Key** -- Obtain from [console.groq.com](https://console.groq.com)
- **GPU (optional)** -- CUDA-compatible GPU accelerates CLIP, Whisper, and embedding computations. The system falls back to CPU automatically.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/safalsingh1/talk-to-data-.git
cd talk-to-data-
```

### 2. Backend Setup

Create and activate a Python virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` includes `git+https://github.com/openai/CLIP.git` which requires Git to be available on your system PATH. If installation fails for CLIP, ensure Git is installed and accessible.

### 3. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

---

## Configuration

### Environment Variables

Copy the example environment file and insert your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your Groq API key:

```env
GROQ_API_KEY=gsk_your_key_here
```

The backend loads this via `python-dotenv` at startup. No other environment variables are required for basic operation.

### Semantic Dictionary (Optional)

Define custom business metric formulas in `semantic_dict.json` at the project root. The LLM will use these exact definitions when users query the corresponding metrics:

```json
{
  "gross_margin": "Gross Margin = (Revenue - COGS) / Revenue * 100",
  "churn_rate": "Churn Rate = Customers Lost / Total Customers at Start of Period * 100"
}
```

These definitions can also be managed via the API at runtime (`POST /api/metrics`).

---

## Running the Application

### Start the Backend Server

```bash
# From the project root, with the virtual environment activated
python api_server.py
```

The FastAPI server starts on `http://localhost:8000`. On first launch, the following models are loaded into memory:

| Model | Size (approx.) | Purpose |
|---|---|---|
| CLIP ViT-B/32 | ~340 MB | Image embedding for RAG |
| all-MiniLM-L6-v2 | ~80 MB | Text embedding for RAG |
| Whisper small | ~460 MB | Audio transcription |

Initial model download may take several minutes depending on network speed. Subsequent launches load from cache.

### Start the Frontend Dev Server

```bash
cd frontend
npm run dev
```

The Vite dev server starts on `http://localhost:5173` (default) with hot module replacement enabled. The frontend proxies API requests to `http://localhost:8000`.

### Verify Connectivity

Open `http://localhost:5173` in a browser. The header bar should display a green connection indicator. If it shows "Offline", confirm the backend is running and accessible on port 8000.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Overview](#overview)
- [Architecture](#architecture)
- [System Architecture Diagram](#system-architecture-diagram)
- [Processing Pipeline](#processing-pipeline)
- [Module Reference](#module-reference)
- [Tech Stack](#tech-stack)
- [API Reference](#api-reference)
- [Frontend Structure](#frontend-structure)
- [Security Considerations](#security-considerations)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

Talk-to-Data bridges the gap between non-technical users and their datasets by providing a natural language interface for data exploration, analysis, and visualization. Rather than writing SQL or Pandas code manually, users upload a dataset (CSV, Excel) or unstructured files (PDF, audio, images, plain text) and ask questions in conversational English. The system interprets intent, generates executable code, runs statistical routines, and returns both narrative explanations and interactive charts.

### Core Capabilities

- **Natural Language to Pandas Translation** -- Converts free-form queries into Pandas expressions with automatic error correction and retry logic.
- **Multi-Agent Intent Routing** -- A hybrid keyword-scoring and LLM-fallback classifier routes queries to specialized processing agents (structured analysis, RAG retrieval, comparison, root cause analysis, breakdown, summary, anomaly detection).
- **Retrieval-Augmented Generation (RAG)** -- Indexes uploaded PDFs, audio transcripts, images, and text files into FAISS vector stores for semantic search-driven Q&A.
- **Automated Statistical Analysis** -- Root cause decomposition, period-over-period comparison, metric breakdown, IQR-based anomaly detection, and auto-summary generation.
- **Interactive Visualization** -- Dynamically selects optimal chart types (bar, line, pie, scatter, heatmap, stacked bar, horizontal bar) and renders them via Recharts on the frontend.
- **PII Detection and Masking** -- Identifies columns containing personally identifiable information through pattern matching on column names and value-level heuristics, with optional masking.
- **Export Pipeline** -- Generates downloadable CSV and PDF reports with embedded charts, data tables, and transparency metadata.
- **Multimodal Input** -- Accepts audio files (transcribed via OpenAI Whisper), PDF documents (text + image extraction), images (indexed via CLIP), and plain text files.

---

## Architecture

The application follows a decoupled client-server architecture with a React single-page application communicating with a FastAPI backend over REST.

```
                       +---------------------+
                       |   React Frontend    |
                       |   (Vite + Recharts) |
                       +---------+-----------+
                                 |
                            HTTP REST
                                 |
                       +---------v-----------+
                       |   FastAPI Backend   |
                       |   (api_server.py)   |
                       +---------+-----------+
                                 |
              +------------------+------------------+
              |                  |                  |
    +---------v------+  +--------v-------+  +-------v--------+
    |  LLM Engine    |  | Analysis Engine|  | Vector DB      |
    |  (Groq API)    |  | (Pandas/NumPy) |  | (FAISS)        |
    +----------------+  +----------------+  +----------------+
              |                                     |
    +---------v------+                    +---------v--------+
    | Visualization  |                    | Embedding Models |
    | Engine         |                    | (CLIP, MiniLM,   |
    | (Matplotlib)   |                    |  Whisper)        |
    +----------------+                    +------------------+
```

### Backend Layers

| Layer | Module | Responsibility |
|---|---|---|
| **API Gateway** | `api_server.py` | Request routing, file upload handling, response serialization, CORS management, static file serving |
| **Intelligence** | `llm_engine.py` | Intent classification, NL-to-Pandas translation, clarification engine, narration agents, metric/dimension detection |
| **Analysis** | `analysis_engine.py` | Root cause analysis, period comparison, segment comparison, metric breakdown, anomaly detection, summary generation, insight ranking |
| **Retrieval** | `vectordb.py` | FAISS index management, vector embedding storage, text/image/audio index search operations |
| **Visualization** | `visualize.py` | Chart type auto-detection, Matplotlib chart generation with dark theme styling, comparison and anomaly chart renderers |
| **Utilities** | `utils.py` | Model loading (CLIP, Sentence-Transformers, Whisper), FAISS index I/O, schema detection, PII detection, data masking |
| **Export** | `export_engine.py` | PDF report generation via ReportLab, CSV export, chart-to-image conversion |

---

## System Architecture Diagram

```
User Query
    |
    v
+----------------------------+
| Intent Router              |
| (Keyword Scoring + LLM)   |
+----------------------------+
    |
    +---> structured -------> NL-to-Pandas -----> Code Execution -----> Chart + Table + Narrative
    |                              |
    |                              +---> Retry on Error (max 2)
    |
    +---> comparison -------> detect_metric_and_dimensions --> compare_periods / compare_segments --> Narration
    |
    +---> root_cause -------> detect_metric_and_dimensions --> root_cause_analysis --> Narration
    |
    +---> breakdown --------> detect_metric_and_dimensions --> breakdown_metric --> Narration
    |
    +---> summary ----------> generate_data_summary --> Narration
    |
    +---> anomaly ----------> detect_anomalies --> Narration + Scatter Chart
    |
    +---> rag --------------> FAISS Semantic Search --> Context Assembly --> LLM Answer Generation
```

---

## Processing Pipeline

### Structured Query Flow

1. **Input** -- User submits a natural language query.
2. **Clarity Check** -- The clarification engine rejects trivially empty or meaningless inputs; all substantive queries pass through.
3. **Intent Classification** -- The router scores the query against keyword dictionaries for eight intent categories, applying weighted scoring with LLM fallback for ambiguous cases.
4. **Code Generation** -- The NL-to-Pandas agent receives the query, column schema, sample data, data types, conversation history, and any defined business metrics. It produces a sandboxed Pandas expression.
5. **Execution** -- The generated code runs within a restricted `eval`/`exec` sandbox that blocks imports, OS access, dunders, and other unsafe operations.
6. **Error Recovery** -- On execution failure, the system sends the error message and original code back to the LLM for correction, repeating up to two times.
7. **Chart Inference** -- Based on result shape, data types, column count, and query keywords, the engine selects the optimal chart type.
8. **Response Assembly** -- The result data is converted to chart-friendly JSON (for Recharts) and table-friendly JSON, paired with an LLM-generated narrative and transparency metadata (generated code, explanation).

### RAG Query Flow

1. **Ingestion** -- Uploaded files are chunked (1000 characters, 200 overlap), embedded via `all-MiniLM-L6-v2` (text) or `ViT-B/32` CLIP (images), and stored in FAISS flat L2 indices alongside CSV-backed metadata.
2. **Retrieval** -- On query, the input is embedded and the top-k nearest neighbors are retrieved from relevant indices (text, audio, image).
3. **Generation** -- Retrieved context chunks are assembled into a prompt and passed to the LLM for answer synthesis with source attribution.

---

## Module Reference

### `llm_engine.py` -- Multi-Agent Intelligence

Contains nine specialized agents, each with its own system prompt and processing logic:

| Agent | Function | Purpose |
|---|---|---|
| Intent Router | `route_intent()` | Classifies queries into one of eight intent categories |
| Clarification Engine | `check_query_clarity()` | Filters trivially empty queries; passes all substantive inputs |
| Data Executor | `nl_to_pandas()`, `nl_to_pandas_with_retry()` | Translates NL to Pandas code with automatic error correction |
| Insight Generator | `generate_explanation()`, `generate_answer_from_pandas()`, `generate_insights()` | Produces natural language summaries of data results |
| Comparison Narrator | `narrate_comparison()` | Narrates period-over-period comparison results |
| Root Cause Narrator | `narrate_root_cause()` | Narrates root cause analysis findings |
| Breakdown Narrator | `narrate_breakdown()` | Narrates metric decomposition results |
| Summary Narrator | `narrate_summary()` | Narrates comprehensive dataset summaries |
| Anomaly Narrator | `narrate_anomalies()` | Narrates anomaly detection results |

### `analysis_engine.py` -- Statistical Analysis

| Function | Description |
|---|---|
| `root_cause_analysis()` | Decomposes metric change across all categorical dimensions; ranks drivers by absolute contribution percentage |
| `compare_periods()` | Computes aggregated statistics (sum, mean, min, max, count) for two period values and calculates deltas |
| `compare_segments()` | Breaks down a metric across all segments of a categorical column with share percentages |
| `breakdown_metric()` | Multi-dimensional decomposition with outlier flagging based on deviation from expected uniform share |
| `detect_anomalies()` | IQR or Z-score based anomaly detection with configurable threshold; reports per-column bounds, counts, directional breakdown |
| `generate_data_summary()` | Produces structured dataset overview including numeric stats, categorical distributions, date ranges, missing data, and skewness detection |
| `rank_insights()` | Scores and ranks a list of insights by magnitude and type-based boosting |

### `vectordb.py` -- Vector Store Management

Manages three independent FAISS indices (`text_index.index`, `audio_index.index`, `image_index.index`) with companion CSV metadata files. Supports:

- Dynamic index creation and incremental updates
- Dimensionality handling (384-dim for text embeddings, 512-dim for CLIP image embeddings)
- Search functions for text-to-text, text-to-image, and image-to-image retrieval

### `visualize.py` -- Visualization Engine

- Intelligent chart type detection based on data shape, column types, row count, and query keywords
- Seven chart types: bar, line, pie, scatter, heatmap, stacked bar, horizontal bar
- Dark-theme Matplotlib styling with custom color palette
- Specialized renderers for comparison charts, root cause driver charts, and anomaly scatter plots
- Sandboxed Pandas code execution with security pattern matching

### `export_engine.py` -- Report Generation

- PDF generation via ReportLab with custom typography, branded color scheme, and structured sections (query, response, chart, data table, transparency details)
- CSV export with UTF-8 encoding

---

## Tech Stack

### Backend

| Component | Technology |
|---|---|
| Web Framework | FastAPI with Uvicorn |
| LLM Provider | Groq (Llama 3.3 70B Versatile) |
| Vision Model | Llama 3.2 90B Vision Preview (via Groq) |
| Text Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Image Embeddings | OpenAI CLIP (`ViT-B/32`, 512-dim) |
| Audio Transcription | OpenAI Whisper (`small` model) |
| Vector Store | FAISS (`IndexFlatL2`) |
| Data Processing | Pandas, NumPy, Scikit-learn, SciPy |
| Chart Generation | Matplotlib (server-side, dark theme) |
| PDF Generation | ReportLab |
| Document Parsing | PyPDF2, LangChain Text Splitters |

### Frontend

| Component | Technology |
|---|---|
| Framework | React 19 with Vite 8 |
| Charting | Recharts 3.8 |
| Icons | Lucide React |
| Styling | Tailwind CSS 4 |
| Build Tool | Vite with PostCSS + Autoprefixer |

---

---

## API Reference

All endpoints are prefixed with `/api`. The backend enforces permissive CORS (`allow_origins=["*"]`) suitable for local development.

### Health and Schema

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Returns `{"status": "ok", "has_data": bool}` |
| `GET` | `/api/schema` | Returns detected schema, column types, and a 5-row preview of the loaded dataset |

### Data Upload

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/upload` | `multipart/form-data` (`file`, optional `sheet_name`) | Upload CSV or Excel file. Returns row/column counts and detected schema |

### RAG File Upload

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/upload/audio` | `multipart/form-data` (`file`) | Transcribes audio via Whisper, chunks, and indexes into FAISS |
| `POST` | `/api/upload/pdf` | `multipart/form-data` (`file`) | Extracts text and images from PDF, indexes both |
| `POST` | `/api/upload/image` | `multipart/form-data` (`file`) | Embeds image via CLIP and indexes |
| `POST` | `/api/upload/text` | `multipart/form-data` (`file`) | Chunks and indexes plain text |
| `GET` | `/api/rag-files` | -- | Lists all uploaded RAG files and index availability |

### Query and Analysis

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/query` | `{"query": str, "routing_mode": str}` | Process a natural language query. `routing_mode` accepts `"Auto-Detect"`, `"Structured (CSV)"`, or `"Unstructured (RAG)"` |
| `POST` | `/api/quick-action` | `{"action": str}` | Execute pre-defined analysis. `action` accepts `"summary"`, `"anomaly"`, or `"insights"` |
| `POST` | `/api/generate-chart` | `{"query": str, "chart_type": str\|null}` | Generate a chart from a natural language description with optional type override |

### Exports

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/export/csv` | Download current dataset as CSV |
| `POST` | `/api/export/pdf` | Generate and download a PDF report from query and response text |

### Metrics and Settings

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/metrics` | Retrieve all defined business metrics |
| `POST` | `/api/metrics` | Add a new business metric definition |
| `POST` | `/api/settings/privacy` | Toggle PII masking on/off |
| `POST` | `/api/clear` | Clear chat history and query cache |

---

## Frontend Structure

```
frontend/src/
    App.jsx              -- Root component: layout, state management, event orchestration
    api.js               -- API client: all backend communication functions
    main.jsx             -- Application entry point
    index.css            -- Design system: tokens, surfaces, components, animations
    components/
        Sidebar.jsx      -- File upload controls, schema display, quick actions, routing, export
        ChatMessage.jsx  -- Message rendering with markdown, charts, tables, trust layer
        ChartRenderer.jsx-- Recharts-based chart components (bar, line, pie, scatter, grouped bar, anomaly)
        DataTable.jsx    -- Tabular data display with column headers and row limiting
        FileUpload.jsx   -- Drag-and-drop file upload with category detection
        SchemaDisplay.jsx-- Dataset schema and column type visualization
        TrustLayer.jsx   -- Transparency panel showing generated Pandas code and explanations
```

---

## Security Considerations

### Code Execution Sandbox

All LLM-generated Pandas code is executed within a restricted sandbox (`visualize.py:execute_pandas_code_safely`). The following patterns are blocked at the string level before evaluation:

- Import statements (`import `, `__import__`)
- OS and system access (`os.`, `sys.`, `subprocess`, `shutil`, `pathlib`)
- Dangerous builtins (`eval(`, `exec(`, `open(`)
- Dunder access (`__`)

The execution environment is limited to a namespace containing only `pd` (Pandas) and `df` (the loaded DataFrame). No other modules, globals, or builtins are accessible.

### PII Protection

The system automatically scans uploaded datasets for potentially sensitive columns using:

1. **Column name pattern matching** against 30+ PII-related regex patterns (SSN, email, phone, credit card, etc.)
2. **Value-level heuristics** sampling the first 20 non-null values to detect email addresses and phone number formats.

When privacy mode is enabled, identified columns are replaced with `"****"` in the working copy sent to the LLM. The original data remains unmodified.

---

## Project Structure

```
talk-to-data-/
    api_server.py            -- FastAPI application with all REST endpoints
    llm_engine.py            -- Multi-agent LLM orchestration (9 specialized agents)
    analysis_engine.py       -- Statistical analysis functions (RCA, comparisons, anomaly detection)
    vectordb.py              -- FAISS vector store management and search
    visualize.py             -- Chart generation engine with dark-theme Matplotlib
    export_engine.py         -- PDF and CSV export pipeline
    utils.py                 -- Model loaders, schema detection, PII detection
    app.py                   -- Legacy Streamlit application (preserved for reference)
    requirements.txt         -- Python dependencies
    semantic_dict.json       -- User-defined business metric definitions
    .env.example             -- Environment variable template
    .gitignore               -- Git ignore rules
    LICENSE                  -- Apache License 2.0
    frontend/
        index.html           -- HTML entry point
        package.json         -- Node.js dependencies and scripts
        vite.config.js       -- Vite bundler configuration
        tailwind.config.js   -- Tailwind CSS configuration
        postcss.config.js    -- PostCSS plugin chain
        src/                 -- React source code (see Frontend Structure)
    vectorstore/             -- FAISS indices and metadata CSVs (generated at runtime)
    images/                  -- Uploaded images (generated at runtime)
    audio/                   -- Uploaded audio files (generated at runtime)
    data_upload/             -- Input source processing utilities
    .devcontainer/           -- Dev container configuration for cloud-based development
```

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](./LICENSE) for the full terms.
