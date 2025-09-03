# 🤖 Smart AI Chatbot with RAG

A powerful AI chatbot that combines general knowledge with your personal documents using Retrieval-Augmented Generation (RAG) technology.

## ✨ Features

- **Dual Mode Chatbot**: Toggle between general knowledge and personal document search
- **Document Processing**: Supports TXT and PDF files
- **Local LLM**: Uses Ollama with llama3.2 model for privacy
- **Vector Database**: ChromaDB for efficient document retrieval
- **Web Interface**: Beautiful Gradio UI
- **Jupyter Integration**: Full notebook support for development

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Ollama installed and running
- llama3.2 model downloaded

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ollama-llm-rag-chatbot.git
   cd ollama-llm-rag-chatbot
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Set up Ollama**
   ```bash
   # Install Ollama (if not already installed)
   brew install ollama
   
   # Start Ollama service
   brew services start ollama
   
   # Download the llama3.2 model
   ollama pull llama3.2
   ```

4. **Add your documents**
   - Place your TXT and PDF files in the `data/` directory
   - The chatbot will automatically process them

### Running the Application

#### Option 1: Jupyter Notebook
```bash
jupyter lab
```
Open `Local AI Chatbot.ipynb` and run the cells.

#### Option 2: Python Script
```bash
python chatbot_ollama_rag_llama32.py
```

#### Option 3: Gradio Interface
```bash
python main.py
```

## 📁 Project Structure

```
ollama-llm-rag-chatbot/
├── data/                    # Your documents (TXT, PDF)
├── chroma_db/              # Vector database (auto-generated)
├── chatbot_ollama_rag_llama32.py  # Main chatbot script
├── Local AI Chatbot.ipynb   # Jupyter notebook
├── main.py                 # Gradio interface
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Supported File Types
- **Text files**: `.txt`
- **PDF files**: `.pdf`
- **Markdown files**: `.md` (with additional setup)

### Model Configuration
The chatbot uses:
- **Embeddings**: OllamaEmbeddings with llama3.2
- **LLM**: Custom SimpleOllamaLLM wrapper
- **Vector Store**: ChromaDB

## 🎯 Usage

1. **Start the application**
2. **Upload or place documents** in the `data/` folder
3. **Ask questions** in the chat interface
4. **Toggle "Use Personal Documents"** to switch between general knowledge and document search

## 🔒 Privacy

- **Local Processing**: All processing happens on your machine
- **No External APIs**: Uses local Ollama instance
- **Document Privacy**: Your documents never leave your system

## 🛠️ Development

### Adding New File Types
To support additional file types, modify the document loading section:

```python
# Add new loaders
from langchain.document_loaders import UnstructuredWordDocumentLoader

# Load Word documents
word_loader = DirectoryLoader('data', glob='**/*.docx', loader_cls=UnstructuredWordDocumentLoader)
word_documents = word_loader.load()
```

### Customizing the Model
Edit the `SimpleOllamaLLM` class in `chatbot_ollama_rag_llama32.py` to modify:
- Model parameters
- API endpoints
- Response formatting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Gradio](https://gradio.app/) for the web interface

## 📞 Support

If you encounter any issues:
1. Check that Ollama is running: `curl http://localhost:11434/api/tags`
2. Verify the llama3.2 model is installed: `ollama list`
3. Check the logs for error messages
4. Open an issue on GitHub

---

**Happy chatting! 🤖💬**
