import os

# Paths
PDF_DIR = "pdfs"
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)  # Create index directory if it doesn't exist

# Chunking
CHUNK_SIZE = 2000
OVERLAP = 200

# Embedding
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # 384 dims, fast, free

# Retrieval
TOP_K = 10  # number of chunks to retrieve per query, this is to give the generator more chances to find the answer

# Generation
LLM_MODEL = "llama3.2:3b" # Large language model for text generation