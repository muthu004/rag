from typing import List, Dict, Any

from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from google import genai
from google.genai import types
EMBEDDING_MODEL = "gemini-embedding-2"
GENERATIVE_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class GeminiEmbedder(Embeddings):
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.client = genai.Client()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return response.embeddings[0].values


class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdfs(pdf_docs: List[Any]) -> str:
        raw_text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    raw_text += page_text
        return raw_text

    @staticmethod
    def chunk_text(text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        return splitter.split_text(text)


def build_vectorstore(pdf_docs: List[Any]) -> FAISS:
    raw_text = DocumentProcessor.extract_text_from_pdfs(pdf_docs)
    text_chunks = DocumentProcessor.chunk_text(raw_text)
    return FAISS.from_texts(texts=text_chunks, embedding=GeminiEmbedder())


class ChatbotController:
    @staticmethod
    def ask_gemini(question: str, context: str, chat_history: List[Dict[str, str]]) -> str:
        client = genai.Client()
        history_str = "\n".join(f"{item['role'].capitalize()}: {item['content']}" for item in chat_history)

        prompt = f"""You are a helpful document assistant. Answer only from the context below.
If the answer is not in the context, say you do not have enough information.

Context:
{context}

Chat History:
{history_str}

Question:
{question}

Answer:"""

        response = client.models.generate_content(model=GENERATIVE_MODEL, contents=prompt)
        return response.text