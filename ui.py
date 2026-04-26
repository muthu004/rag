import logging
import os

import streamlit as st
from dotenv import load_dotenv

from rag import ChatbotController, build_vectorstore


logger = logging.getLogger(__name__)


def initialize_ui_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Document Repository")
        pdf_docs = st.file_uploader("Upload reference PDFs:", accept_multiple_files=True, type=["pdf"])

        if st.button("Process & Index", type="primary"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return

            with st.spinner("Processing Documents..."):
                try:
                    st.session_state.vectorstore = build_vectorstore(pdf_docs)
                    st.success("Indexing complete. The assistant is ready.")
                except Exception:
                    logger.exception("Pipeline failure")
                    st.error("A critical error occurred during document processing. Check logs for details.")


def render_chat_interface() -> None:
    user_question = st.chat_input("Ask a question about the processed documents...")
    if not user_question:
        return

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            st.warning("Upload and process a document in the sidebar first.")
            return

        try:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
            semantic_matches = retriever.invoke(user_question)
            context_str = "\n\n---\n\n".join(doc.page_content for doc in semantic_matches)
            response_text = ChatbotController.ask_gemini(user_question, context_str, st.session_state.chat_history)

            st.write(response_text)
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"Error resolving query: {e}")


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title="Intelligent Document Assistant", page_icon="📚", layout="wide")
    initialize_ui_state()

    st.title(" Intelligent Document Assistant")
    st.markdown("Unlock insights from your PDFs seamlessly using Google GenAI and FAISS.")

    if not os.environ.get("GEMINI_API_KEY"):
        st.error("Set GEMINI_API_KEY in your environment or .env file.")
        st.stop()

    render_sidebar()
    render_chat_interface()