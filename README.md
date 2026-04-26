# Intelligent Document Assistant

This project is a PDF question-answering app built with Retrieval-Augmented Generation (RAG). Users upload documents, the app chunks and embeds the text, retrieves the most relevant passages with FAISS, and asks Gemini to answer from that context.

## Tech Stack

- Python
- Streamlit
- Google GenAI (`google-genai`)
- FAISS (`faiss-cpu`)
- LangChain text splitters
- PyPDF
- python-dotenv

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Copy [.env.example](.env.example) to `.env` and add your `GEMINI_API_KEY`
4. Run the app:
	```bash
	streamlit run app.py
	```
