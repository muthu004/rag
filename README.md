# Intelligent Document Assistant

> A Retrieval-Augmented Generation (RAG) application that lets you have conversational Q&A sessions with your PDF documents, powered by Google Gemini and FAISS.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?logo=google&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

**Intelligent Document Assistant** allows users to upload one or more PDF documents and ask natural-language questions against their content. The application:

1. **Extracts** text from uploaded PDFs using PyPDF.
2. **Chunks** the text into overlapping segments with LangChain's `RecursiveCharacterTextSplitter`.
3. **Embeds** each chunk via the `gemini-embedding-2` model and indexes them in a FAISS vector store.
4. **Retrieves** the top-*k* semantically relevant passages for each user query.
5. **Generates** a grounded answer using `gemini-2.5-flash`, constrained to the retrieved context.

---

## Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────────┐
│              Streamlit UI (ui.py)         │
│  ┌─────────────┐      ┌────────────────┐  │
│  │  Sidebar    │      │  Chat Interface│  │
│  │ PDF Upload  │      │  Q&A Display   │  │
│  └──────┬──────┘      └───────┬────────┘  │
└─────────┼────────────────────┼────────────┘
          │                    │
          ▼                    ▼
  build_vectorstore()    ask_gemini()
  ┌──────────────┐       ┌───────────────┐
  │  PyPDF       │       │  FAISS        │
  │  Text Chunk  │──────▶│  Retriever    │
  │  Gemini Embed│       │  (top-k docs) │
  └──────────────┘       └──────┬────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Gemini 2.5    │
                        │ Flash LLM     │
                        └───────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| PDF Parsing | [PyPDF](https://pypdf.readthedocs.io/) |
| Text Splitting | [LangChain](https://python.langchain.com/) `RecursiveCharacterTextSplitter` |
| Embeddings | Google Gemini (`gemini-embedding-2`) |
| Vector Store | [FAISS](https://faiss.ai/) (`faiss-cpu`) |
| LLM | Google Gemini (`gemini-2.5-flash`) |
| Configuration | [python-dotenv](https://github.com/theskumar/python-dotenv) |

---

## Prerequisites

- Python 3.9 or higher
- A [Google AI Studio](https://aistudio.google.com/) API key with access to the Gemini models

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/muthu004/rag.git
cd rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

### 5. Run the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

1. **Upload PDFs** — Use the sidebar to upload one or more PDF files.
2. **Process & Index** — Click **Process & Index** to extract, chunk, embed, and index the documents.
3. **Ask Questions** — Type your question in the chat input and receive a context-grounded answer from Gemini.

---

## Project Structure

```
rag/
├── app.py              # Entry point
├── ui.py               # Streamlit UI components
├── rag.py              # RAG pipeline (embedding, retrieval, generation)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md
```

---

## Configuration

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key used for both embeddings and generation |

The following constants in `rag.py` can be adjusted to tune retrieval quality:

| Constant | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `gemini-embedding-2` | Embedding model name |
| `GENERATIVE_MODEL` | `gemini-2.5-flash` | Generative model name |
| `CHUNK_SIZE` | `1000` | Maximum characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlapping characters between consecutive chunks |
