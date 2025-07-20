import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = "uploads"
INDEX_FOLDER = "faiss_indexes"
MEMORY_FOLDER = "memory_graphs"
INTERMEDIATE_RESULTS_FOLDER = "intermediate_results"

# Ensure directories exist (optional, can be handled in main app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)
os.makedirs(MEMORY_FOLDER, exist_ok=True)
os.makedirs(INTERMEDIATE_RESULTS_FOLDER, exist_ok=True)

# Config
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
genai_api_key = os.getenv("GOOGLE_API_KEY")
emb_model = "models/embedding-001"
llm_model = "gemini-1.5-flash"

# CHUNKING STRATEGY
CHUNK_SIZE = 720       # characters per chunk (try 500–1000 depending on your PDFs)
CHUNK_OVERLAP = 200    # overlap between chunks 