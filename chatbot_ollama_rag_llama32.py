# chatbot_ollama_rag_llama32_dual.py

# Required installations (run once):
# pip install langchain chromadb pypdf unstructured gradio openai

from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import Field
import gradio as gr
import openai

# Step 1: Load documents from the "data" directory


# Load documents from the "data" directory (TXT and PDF files)
print("Loading documents...")
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

# Load text files
txt_loader = DirectoryLoader('data', glob='**/*.txt')
txt_documents = txt_loader.load()

# Load PDF files
pdf_loader = DirectoryLoader('data', glob='**/*.pdf', loader_cls=PyPDFLoader)
pdf_documents = pdf_loader.load()

# Combine all documents
documents = txt_documents + pdf_documents
print(f"✅ Loaded {len(txt_documents)} text files and {len(pdf_documents)} PDF files")
print(f"✅ Total documents: {len(documents)}")

# Step 2: Create embeddings and store them in ChromaDB
print("Generating embeddings...")
embedding_model = OllamaEmbeddings(model='llama3.2')
vectordb = Chroma.from_documents(documents, embedding=embedding_model, persist_directory="./chroma_db")
vectordb.persist()

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

# Step 5: Define chatbot logic with toggle


# Step 5: Define chatbot logic with toggle
def chatbot(query, use_personal_data):
    if use_personal_data:
        return qa_pipeline.run(query)
    else:
        return llm._call(query)

# Embed logo (you can also use your local image path or base64 string)
logo_path = "D:/study/Projects/image.png"  # Your local image path

with gr.Blocks(title="Smart AI Chatbot", css="""
    #main-col {
        max-width: 600px;
        margin: 0 auto !important;
        padding-top: 20px;
    }
    .gr-button {
        max-width: 200px;
        margin: 0 auto;
    }
    .gr-textbox textarea {
        font-size: 14px;
    }
    .center-image img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
""") as demo:

    with gr.Column(elem_id="main-col"):
        gr.Image(value=logo_path, width=120, show_label=False, container=False, elem_classes=["center-image"])

        gr.Markdown("## 🤖 Smart AI Chatbot", elem_id="title")
        gr.Markdown("Ask anything — toggle between general knowledge and your own data!", elem_id="subtitle")

        chatbot_ui = gr.Chatbot(label="💬 Chat History")
        input_box = gr.Textbox(label="💬 Your Question", placeholder="Type your message here...", lines=1)
        use_personal = gr.Checkbox(label="📚 Use Personal Documents?")
        btn = gr.Button("🚀 Submit", size="sm")
        state = gr.State([])

        def chat_handler(user_input, use_personal, chat_history):
            if use_personal:
                bot_reply = qa_pipeline.run(user_input)
            else:
                bot_reply = llm._call(user_input)

            chat_history.append((user_input, bot_reply))
            return chat_history, chat_history

        btn.click(fn=chat_handler, inputs=[input_box, use_personal, state], outputs=[chatbot_ui, state])

demo.launch()