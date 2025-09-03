# chatbot_ollama_rag_llama32_dual.py - CORRECTED VERSION

# All packages are already installed in your virtual environment
# No need for pip install - just import directly

# CORRECTED IMPORTS (matching your actual working code):
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import Field
import gradio as gr
import openai

print("✅ All imports successful!")

# Step 1: Load documents from the "data" directory
print("Loading documents...")
loader = DirectoryLoader('data', glob='**/*.txt')
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents")

# Step 2: Create embeddings and store them in ChromaDB
print("Generating embeddings...")
embedding_model = OllamaEmbeddings(model='llama3.2')
vectordb = Chroma.from_documents(documents, embedding=embedding_model, persist_directory="./chroma_db")
vectordb.persist()
print("✅ Embeddings created and stored")

# Step 3: Define custom LLM wrapper for llama3.2 using completion endpoint
class SimpleOllamaLLM(LLM):
    model: str = Field(default="llama3.2", description="Model name to use.")
    base_url: str = Field(default="http://localhost:11434/v1", description="Base URL for Ollama API.")
    api_key: str = Field(default="ollama", description="Dummy API key for compatibility.")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None) -> str:
        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.completions.create(
            model=self.model,
            prompt=prompt,
            stop=stop,
        )
        return response.choices[0].text.strip()

    @property
    def _llm_type(self) -> str:
        return "simple_ollama"

# Step 4: Initialize LLM and QA pipeline
llm = SimpleOllamaLLM(model="llama3.2")
retriever = vectordb.as_retriever()
qa_pipeline = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
print("✅ LLM and QA pipeline initialized")

# Step 5: Define chatbot logic with toggle
def chatbot(query, use_personal_data):
    if use_personal_data:
        return qa_pipeline.run(query)
    else:
        return llm._call(query)

print("✅ Chatbot function defined - ready to use!")

# Test the chatbot
test_query = "Hello, how are you?"
print(f"\n🧪 Testing chatbot with: '{test_query}'")
try:
    response = chatbot(test_query, use_personal_data=False)
    print(f"✅ Response: {response}")
except Exception as e:
    print(f"❌ Error: {e}")
