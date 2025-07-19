
import os
import glob
import fitz  # PyMuPDF
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_DIR = "chroma_db"
PDF_DIR = "resources"

def extract_pdf_pages(file_path):
    print ('_______________')
    print ('file_path = ',file_path)
    """Extract text per page from a PDF using PyMuPDF"""
    doc = fitz.open(file_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        metadata = {
            "source": os.path.basename(file_path),
            "page": i + 1  # human-readable (1-indexed)
        }
        pages.append(Document(page_content=text, metadata=metadata))
    return pages

def load_documents():
    """Load and parse all PDFs into Document objects with metadata"""
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    all_docs = []
    for pdf in pdf_files:
        pages = extract_pdf_pages(pdf)
        all_docs.extend(pages)
    return all_docs

def build_chroma_index(embedding):
    """Create Chroma DB from documents if not already created"""
    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        print("[✅] Chroma DB already exists. Loading from disk...")
        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    else:
        print("[📚] Creating Chroma DB from PDF resources...")
        documents = load_documents()

        # Split long documents into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=CHROMA_DIR
        )
        vectordb.persist()
        print("[💾] Chroma DB created and saved.")
    return vectordb
