# ====================================
# 📌 RAG Setup: Vector Database + PDF Processing
# ====================================

# ---- Install Packages ----
# !pip install -q langchain langchain-community langchain-pinecone langchain-google-genai tavily-python pypdf sentence-transformers

import os
from google.colab import userdata, files

# LangChain utilities
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Embeddings + Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ---------------------------
# 🔑 Load API Keys
# ---------------------------
GOOGLE_KEY = userdata.get("GOOGLE_API_KEY")
PINECONE_KEY = userdata.get("PINECONE_API_KEY")
TAVILY_KEY = userdata.get("TAVILY_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_KEY or ""
os.environ["PINECONE_API_KEY"] = PINECONE_KEY or ""
os.environ["TAVILY_API_KEY"] = TAVILY_KEY or ""

print("🔑 Keys Loaded:", {"Gemini": bool(GOOGLE_KEY), "Pinecone": bool(PINECONE_KEY), "Tavily": bool(TAVILY_KEY)})

# ---------------------------
# 📄 Upload & Process PDF
# ---------------------------
uploaded_file = files.upload()
pdf_file = list(uploaded_file.keys())[0]

loader = PyPDFLoader(pdf_file)
pages = loader.load()
print(f"📄 Loaded {len(pages)} pages from {pdf_file}")

# Split content into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
chunks = splitter.split_documents(pages)
print(f"✂️ Split into {len(chunks)} chunks")

# ---------------------------
# 🧠 Embeddings + Pinecone Setup
# ---------------------------
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE_KEY)
INDEX_NAME = "rag-course-index"

# Create new index if not exists
if INDEX_NAME not in [x["name"] for x in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Created Pinecone index: {INDEX_NAME}")

vector_db = PineconeVectorStore.from_documents(chunks, embedding=emb, index_name=INDEX_NAME)
print("📦 Documents stored in Pinecone successfully")
