
# Nigeria Data Protection RAG System

A question-answering system for Nigeria's data protection documents. Answers questions using ONLY the provided documents and abstains when information is not found.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Documents
Place these PDFs in the `pdfs/` folder:
- ndpa-2023.pdf (Nigeria Data Protection Act 2023)
- gaid-2025.pdf (General Application and Implementation Directive 2025)
- ndpr-2019.pdf (Nigeria Data Protection Regulation 2019)
- ndpr-if-2020.pdf (NDPR Implementation Framework 2020)

### 3. Ingest Documents
```bash
python ingest.py
```
This creates the FAISS index and metadata in the `index/` folder.

### 4. Answer Questions
```bash
python answer.py --questions questions.json --out answers.json
```

### 5. Run Evaluation
```bash
python evaluate.py
```

## Architecture

### Overview
```
PDFs → Chunking → Embeddings → FAISS Index
                ↓
Question → Retrieve Chunks → LLM → JSON Answer
```

### Components

**ingest.py**
- Extracts text from PDFs using pdfplumber
- Chunks text into segments (2000 chars, 200 overlap)
- Creates embeddings using BAAI/bge-small-en-v1.5
- Builds FAISS index for similarity search

**answer.py**
- Embeds the question
- Retrieves top-k similar chunks from FAISS
- Sends context + question to LLM (qwen2.5:7b by default)
- Validates citations against full document texts
- Outputs structured JSON

**evaluate.py**
- Runs 25 test questions
- Verifies citations verbatim
- Reports success/failure rates

## Configuration (config.py)

| Setting | Value | Purpose |
|---------|-------|---------|
| CHUNK_SIZE | 2000 | Text chunk size |
| OVERLAP | 200 | Overlap between chunks |
| EMBEDDING_MODEL | BAAI/bge-small-en-v1.5 | Embedding model (384 dims) |
| TOP_K | 10 | Number of chunks to retrieve |
| LLM_MODEL | qwen2.5:7b | Local LLM for generation |

## Output Format

```json
{
  "id": "q1",
  "answer": "The penalty is N10,000,000 or 2% of annual revenue.",
  "abstained": false,
  "citations": [
    {
      "doc_id": "ndpa-2023",
      "page": 32,
      "quote": "ordering the data controller... to pay a penalty"
    }
  ]
}
```

## File Structure

```
.
├── pdfs/               # Source PDF documents
├── index/              # FAISS index and metadata
├── ingest.py           # Document ingestion
├── answer.py           # Question answering
├── evaluate.py         # Evaluation harness
├── config.py           # Configuration
├── questions.json      # Sample questions
├── answers.json        # Output answers
└── requirements.txt    # Python dependencies
```

## Known Issues

1. **Non-JSON Outputs**: Sometimes the LLM returns text instead of JSON for complex questions (q2, q8, q12, q18). The system currently defaults to abstaining.

2. **Retrieval Gaps**: Some questions (q1, q6) don't retrieve the right chunks, causing abstentions when answers exist.

3. **Multi-Document Comparison**: Questions requiring document comparison sometimes only cite one document or abstain.

4. **Unanswerable Questions**: ~40% of unanswerable questions are incorrectly answered (q22, q23).

## What I'd Improve

1. **Retrieval**: Try smaller chunks (1000 chars) for better precision
2. **Prompt**: Add explicit multi-document comparison instructions
3. **Error Handling**: Extract JSON from LLM responses even when wrapped in text
4. **Validation**: Use fuzzy matching for citations with minor differences
5. **Caching**: Cache embeddings to speed up repeated runs

## Tools Used

- **Ingestion**: pdfplumber, sentence-transformers, faiss-cpu
- **LLM**: Ollama with qwen2.5:7b
- **Evaluation**: Custom Python harness

## Use of AI

I used an AI coding assistant through the development for:
- Code scaffolding (ingest.py, answer.py, test_ingest_retrieval.py)
- Generating test questions (questions.json, eval_questions.json)
- Building the evaluation harness (evaluate.py)
- Drafting documentation (README.md, DESIGN_NOTE.md)
- Debugging assistance and code review

All architectural decisions, retrieval design, prompt engineering, citation validation logic, evaluation metrics, and production recommendations were made independently. The AI was used as a productivity tool, not a substitute for understanding. I fully understand and can explain the components of the system.

## Cost

- **Local**: ~$0 (runs entirely on the machine)
- **Cloud estimate**: ~$0.001 per question (embedding + generation)
