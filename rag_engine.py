import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup

# Load API key from .env file
load_dotenv()

def load_documents(file_path: str):
    """Step 1: Load document based on file type"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents

def load_url(url: str):
    """Load content from a website URL using requests + BeautifulSoup"""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    doc = Document(
        page_content=text,
        metadata={"source": url}
    )
    return [doc]

def chunk_documents(documents):
    """Step 2: Split documents into small chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """Steps 3 & 4: Embed chunks using free HuggingFace model"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def build_qa_chain(vector_store):
    """Steps 5-8: Build the retrieval + LLM chain"""

    prompt_template = """
    You are a helpful assistant. Use the context below to answer the question.
    If the answer is not in the context, say "I couldn't find that in the documents."

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}

def ask_question(qa_chain, question: str):
    """Run a question through the RAG pipeline — returns full answer"""
    chain = qa_chain["chain"]
    retriever = qa_chain["retriever"]
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": sources
    }

def stream_answer(qa_chain, question: str):
    """Stream answer word by word — for typing animation"""
    chain = qa_chain["chain"]
    for chunk in chain.stream(question):
        yield chunk