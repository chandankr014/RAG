# Advanced RAG QA Chatbot

A sophisticated PDF question-answering chatbot that implements multiple advanced Retrieval-Augmented Generation (RAG) techniques for improved accuracy and reasoning capabilities.

## 🚀 Features

### Core Functionality
- **PDF Upload & Indexing**: Upload PDF documents and automatically create searchable vector embeddings
- **Smart Question Answering**: Ask questions about your uploaded documents
- **Source Attribution**: See exactly which pages and sections your answers come from

### Advanced RAG Techniques

#### 1. 🤖 LLM-Powered Decomposition
- **What it does**: Breaks complex questions into simpler sub-questions
- **How it works**: Uses an LLM to decompose complex queries into 2-4 focused sub-questions, answers each separately, then synthesizes a comprehensive response
- **Best for**: Multi-part questions, analytical queries, questions requiring multiple perspectives

#### 2. 🧠 Chain-of-Thought (CoT) Prompting
- **What it does**: Uses step-by-step reasoning to answer questions
- **How it works**: Guides the LLM through explicit reasoning steps, making the thinking process transparent and improving accuracy
- **Best for**: Logical reasoning questions, problem-solving, questions requiring analysis

#### 3. 🏗️ Graph-Based Reasoning
- **What it does**: Builds and queries a knowledge graph of entities and relationships
- **How it works**: Extracts named entities from documents, creates relationships between them, and uses graph traversal for enhanced reasoning
- **Best for**: Questions about relationships between entities, complex document analysis, multi-entity queries

#### 4. 🔄 Auto Technique Selection
- **What it does**: Automatically chooses the best technique based on question complexity
- **How it works**: Tries techniques in order of sophistication, falling back gracefully if needed
- **Best for**: Users who want optimal results without manual technique selection

## 🛠️ Installation

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd QA-Chatbot

# Run the installation script
python install_dependencies.py

# Set up your environment variables
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Start the application
python app.py
```

### Manual Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install spaCy English model (for graph reasoning)
python -m spacy download en_core_web_sm

# Set up environment variables
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### Verify Installation
```bash
# Test that everything is working
python test_setup.py
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_generative_ai_api_key
SECRET_KEY=your_flask_secret_key_optional
```

### Centralized Configuration (`config.py`)
All configuration variables (such as folder paths, chunking strategy, model names, and file size limits) are now managed in `config.py`.

- **Edit `config.py`** to change upload/index/memory/intermediate results folder names, chunk size, overlap, or model settings.
- `config.py` loads environment variables from `.env` and ensures all required directories exist.
- The main app (`app.py`) imports all configuration from `config.py` and uses them globally.

Example snippet from `config.py`:
```python
UPLOAD_FOLDER = "uploads"
INDEX_FOLDER = "faiss_indexes"
MEMORY_FOLDER = "memory_graphs"
INTERMEDIATE_RESULTS_FOLDER = "intermediate_results"
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
genai_api_key = os.getenv("GOOGLE_API_KEY")
emb_model = "models/embedding-001"
llm_model = "gemini-1.5-flash"
CHUNK_SIZE = 720
CHUNK_OVERLAP = 200
```

No configuration is hardcoded in `app.py`—all such values are imported from `config.py`.

### API Key Setup
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## 📖 Usage

### Web Interface
1. Open your browser to `http://localhost:5000`
2. Upload a PDF document (max 25MB)
3. Wait for indexing to complete
4. Select your preferred RAG technique from the dropdown
5. Ask questions about your document

### Technique Selection Guide

| Technique | Best For | Example Questions |
|-----------|----------|-------------------|
| **Auto** | General use, complex questions | "What are the main challenges and solutions discussed?" |
| **Decomposition** | Multi-part questions | "Compare the advantages and disadvantages of different approaches" |
| **Chain-of-Thought** | Logical reasoning | "How does the process work step by step?" |
| **Graph Reasoning** | Entity relationships | "What is the relationship between X and Y?" |
| **Basic RAG** | Simple factual questions | "What is the definition of X?" |

### API Endpoints

#### Upload PDF
```bash
POST /upload
Content-Type: multipart/form-data

file: your_document.pdf
```

#### Ask Question
```bash
POST /ask-question/
Content-Type: application/json

{
  "question": "Your question here",
  "technique": "auto"  // auto, decomposition, cot, graph, basic
}
```

#### Get Available Techniques
```bash
GET /techniques
```

## 🏗️ Architecture

### File Structure
```
QA-Chatbot/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── install_dependencies.py # Installation helper
├── README.md             # This file
├── .env                  # Environment variables (create this)
├── uploads/              # Uploaded PDF files
├── faiss_indexes/        # Vector embeddings
├── memory_graphs/        # Knowledge graph data
├── static/               # Frontend assets
│   ├── css/
│   └── js/
└── templates/            # HTML templates
```

### Key Components

#### 1. Document Processing
- **PDF Parsing**: Uses PyMuPDF for text extraction
- **Chunking**: Intelligent text splitting with overlap
- **Embedding**: Google's embedding-001 model for vector creation
- **Indexing**: FAISS for fast similarity search

#### 2. RAG Techniques Implementation

**LLM Decomposition**:
```python
def decompose_question(question: str) -> List[str]:
    # Uses LLM to break complex questions into sub-questions
    # Returns list of simpler questions to answer separately
```

**Chain-of-Thought**:
```python
def get_cot_chain():
    # Creates prompts that guide step-by-step reasoning
    # Makes thinking process explicit and traceable
```

**Graph Reasoning**:
```python
class KnowledgeGraph:
    # Extracts entities using spaCy
    # Builds relationship graph
    # Queries graph for enhanced reasoning
```

#### 3. Frontend Features
- **Technique Selector**: Dropdown to choose RAG method
- **Enhanced Display**: Technique-specific styling and information
- **Progress Indicators**: Real-time feedback during processing
- **Source Attribution**: Clear citation of document sources

## 🔍 Advanced Features

### Knowledge Graph Persistence
- Graph data is automatically saved to `memory_graphs/`
- Entities and relationships persist between sessions
- Improves performance for repeated queries

### Intermediate Results Tracking
- **Complete Process Logging**: Every step of each RAG technique is logged
- **Structured Storage**: Results stored in `intermediate_results/` folder
- **Session-based Organization**: Each question gets a unique session ID
- **Step-by-step Analysis**: See exactly what happened during processing

#### Intermediate Results Structure
```
intermediate_results/
├── {session_id}/
│   ├── decomposition/
│   │   ├── question_decomposition.json
│   │   ├── sub_question_1_result.json
│   │   ├── sub_question_2_result.json
│   │   └── final_synthesis.json
│   ├── cot/
│   │   ├── document_retrieval.json
│   │   └── reasoning_process.json
│   ├── graph/
│   │   ├── vector_search.json
│   │   ├── entity_extraction.json
│   │   ├── graph_query.json
│   │   └── final_reasoning.json
│   └── basic/
│       ├── document_retrieval.json
│       └── qa_result.json
```

#### Viewing Intermediate Results
```bash
# List all sessions
python view_intermediate_results.py --list

# View latest session
python view_intermediate_results.py --latest

# View specific session
python view_intermediate_results.py --session {session_id}
```

#### API Endpoints for Results
```bash
# List all sessions
GET /intermediate-results

# Get specific session results
GET /intermediate-results/{session_id}
```

#### What Gets Logged for Each Technique

**🔀 LLM Decomposition:**
- Original question and decomposed sub-questions
- Each sub-question's retrieved documents
- Individual answers for each sub-question
- Final synthesis process and result

**🧠 Chain-of-Thought:**
- Document retrieval results
- Step-by-step reasoning process
- Final answer with reasoning steps

**🏗️ Graph Reasoning:**
- Vector search results
- Entity extraction from documents
- Graph query and traversal
- Final reasoning with combined context

**📚 Basic RAG:**
- Document retrieval
- QA chain results
- Strict vs. non-strict mode usage

### Fallback Mechanisms
- Graceful degradation if spaCy model is unavailable
- Automatic technique switching if one fails
- Error handling with user-friendly messages

### Performance Optimizations
- Threaded PDF processing for large documents
- Efficient vector similarity search with FAISS
- Cached embeddings and graph data

## 🧪 Testing Different Techniques

### Example Questions to Try

**For Decomposition**:
- "What are the main challenges, solutions, and future directions discussed in this document?"
- "Compare and contrast the different methodologies presented"

**For Chain-of-Thought**:
- "How does the forecasting process work from start to finish?"
- "What are the logical steps to implement this solution?"

**For Graph Reasoning**:
- "What is the relationship between Model A and Model B?"
- "How do the different components interact with each other?"

**For Basic RAG**:
- "What is the definition of forecasting?"
- "What are the key metrics mentioned?"

## 🐛 Troubleshooting

### Common Issues

**OpenMP Runtime Conflict**:
If you see an error like "OMP: Error #15: Initializing libomp140.x86_64.dll, but found libiomp5md.dll already initialized", this is a common issue with FAISS and other libraries that use OpenMP.

**Solutions**:
1. **Automatic fix**: The app now includes the fix automatically
2. **Manual fix**: Set environment variable before running:
   ```bash
   # Windows
   set KMP_DUPLICATE_LIB_OK=TRUE
   python app.py
   
   # Unix/Linux/Mac
   export KMP_DUPLICATE_LIB_OK=TRUE
   python app.py
   ```
3. **Use provided scripts**:
   ```bash
   # Windows
   run_app.bat
   
   # Unix/Linux/Mac
   chmod +x run_app.sh
   ./run_app.sh
   ```

**spaCy Model Not Found**:
```bash
python -m spacy download en_core_web_sm
```

**API Key Issues**:
- Ensure your Google API key is valid and has sufficient quota
- Check that the key is properly set in `.env` file

**Large File Processing**:
- Files are limited to 25MB for performance
- Processing time scales with document size
- Progress indicators show real-time status

**Memory Issues**:
- Graph data is automatically cleaned up
- FAISS indexes are optimized for memory usage
- Consider using smaller chunk sizes for very large documents

## 🔮 Future Enhancements

- [ ] Multi-document support
- [ ] Conversation memory across sessions
- [ ] Advanced graph visualization
- [ ] Custom prompt templates
- [ ] Export/import of knowledge graphs
- [ ] Real-time collaboration features

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the error messages in the console
3. Open an issue with detailed information about your problem

---

**Happy Question Answering! 🎉** 