import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# === Load environment variables ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # 👈 Securely load from .env

# === Define paths ===
FAISS_INDEX_PATH = "vectorstore/faiss_index"


class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.vectorstore = self._load_vectorstore()
        self.llm = self._load_llm()
        self.qa_chain = self._setup_qa_chain()

    def _load_vectorstore(self):
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Please run `python ingest.py` first to create the vector store."
            )
        print(f"✅ Loading FAISS index from: {FAISS_INDEX_PATH}")
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def _load_llm(self):
        if not GROQ_API_KEY:
            raise ValueError("❌ GROQ_API_KEY not found in .env file.")
        print("✅ Loading LLM from Groq API (llama3-8b-8192)...")
        return ChatGroq(
            temperature=0.1,
            model_name="llama3-8b-8192",
            api_key=GROQ_API_KEY  # ✅ Pass the loaded key
        )

    def _setup_qa_chain(self):
        prompt_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def run(self, query: str):
        print(f"\n🔍 Running query: {query}")
        result = self.qa_chain.invoke({"query": query})
        answer = result["result"]
        source_documents = result.get("source_documents", [])
        return answer, source_documents


# === Script Entry Point ===
if __name__ == "__main__":
    try:
        rag_pipeline = RAGPipeline()
        test_query = "What is the policy on vacation days?"
        answer, sources = rag_pipeline.run(test_query)

        print("\n✅ Answer:\n", answer)

        print("\n📚 Sources:")
        for source in sources:
            print(f"- {os.path.basename(source.metadata['source'])} (Page {source.metadata['page']})")
            print(f"  → Snippet: {source.page_content[:200]}...\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure to run `python ingest.py` to create the FAISS index.")

    except ValueError as e:
        print(f"\n❌ API Key Error: {e}")
        print("Please check that your `.env` file contains: GROQ_API_KEY=your-key-here")
