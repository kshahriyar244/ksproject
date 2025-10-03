# ====================================
# 📌 RAG Query System: Smart Q&A
# ====================================

import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults

# ---------------------------
# 🔑 Load Keys
# ---------------------------
G_KEY = os.environ.get("GOOGLE_API_KEY", "")
P_KEY = os.environ.get("PINECONE_API_KEY", "")
T_KEY = os.environ.get("TAVILY_API_KEY", "")

# ---------------------------
# ⚡ Initialize LLM
# ---------------------------
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=G_KEY)

# Vector DB (use same index as setup)
INDEX = "rag-course-index"
vector_store = PineconeVectorStore(index_name=INDEX, embedding=None)

# ---------------------------
# 📋 Prompt Template
# ---------------------------
qa_template = """
You are a course assistant. Use the given context to answer questions.
If the answer is missing, reply: "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=qa_template)
qa_chain = LLMChain(llm=model, prompt=prompt)

# Tavily search tool
web_search = TavilySearchResults()

# ---------------------------
# 🤖 Smart Q&A Function
# ---------------------------
def query_rag(question: str, top_k: int = 3, threshold: float = 0.4):
    """Try Pinecone first, fallback to Tavily search if weak results."""
    matches = vector_store.similarity_search_with_score(question, k=top_k)

    if matches and matches[0][1] >= threshold:
        context_text = "\n\n".join([m[0].page_content for m in matches])
        print(f"✅ Answered from Pinecone | score={matches[0][1]:.2f}")
        return qa_chain.invoke({"context": context_text, "question": question})["text"]

    print("🌍 Answered from Tavily Web Search")
    web_results = web_search.invoke({"query": question})

    summarize_prompt = f"""
    Summarize the following search results to answer clearly:

    Question: {question}

    Results:
    {web_results}
    """
    return model.invoke(summarize_prompt).content

# ---------------------------
# 🧪 Example Queries
# ---------------------------
print(query_rag("List all courses available"))
print(query_rag("What are the features in Python 3.12?"))
