import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama
from config import INDEX_DIR, EMBEDDING_MODEL, TOP_K, LLM_MODEL


# Clean document ID
def clean_doc_id(doc_id):
    if not doc_id:
        return ""
    if '|' in doc_id:
        doc_id = doc_id.split('|')[0].strip()
    if doc_id in ["...", "NA", "N/A", ""]:
        return ""
    return doc_id


# Load FAISS index and metadata
def load_index():
    index = faiss.read_index(f"{INDEX_DIR}/faiss_index.bin")
    with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    try:
        with open(f"{INDEX_DIR}/full_texts.pkl", "rb") as f:
            full_texts = pickle.load(f)
    except FileNotFoundError:
        print("ERROR: full_texts.pkl not found.")
        full_texts = {}
    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, metadata, full_texts, model


# Retrieve relevant documents
def retrieve(query, model, index, metadata, top_k=TOP_K):
    emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(emb.astype(np.float32), top_k)    # Search the FAISS index for the most similar chunks
    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(metadata[idx])
    return results


# Validate citation: Check if doc_id exists and quote is not empty
def validate_citation(citation, full_texts):
    raw_doc = citation.get("doc_id")
    quote = citation.get("quote")
    page = citation.get("page")
    
    if not raw_doc or not quote:
        return False
    
    if not page:
        return False

    # Clean doc_id
    doc_id = clean_doc_id(raw_doc)
    if doc_id in ["...", "NA", "N/A", ""]:
        return False
    
    # Check doc_id exists
    if doc_id not in full_texts:
        for key in full_texts.keys():
            if doc_id.lower() in key.lower() or key.lower() in doc_id.lower():
                citation["doc_id"] = key
                return True
        return False
    
    return True


# Generate answer from context
def generate_answer(question, contexts, model_name=LLM_MODEL):
    context_text = "\n\n".join([
        f"[Document: {c['doc_id']} | Page: {c['page']}]\n{c['text']}"   # Format the retrieved text chunks with their document IDs and page numbers
        for c in contexts
    ])

    # System prompt
    system_prompt = """You are a precise assistant. Answer the user's question based ONLY on the provided text excerpts.

CRITICAL RULES:
1. The doc_id MUST be exactly one of: ndpa-2023, gaid-2025, ndpr-2019, ndpr-if-2020
2. The quote MUST be copied VERBATIM from the source text
3. Include the page number where the quote appears
4. If the answer is not present, set "abstained" to true

Output strictly in this JSON format:
{
  "answer": "your answer here",
  "abstained": false,
  "citations": [
    {"doc_id": "ndpa-2023", "page": 32, "quote": "verbatim text from the source"}
  ]
}

If you abstain, citations must be an empty list.
Only output JSON, nothing else."""

    # User prompt
    user_prompt = f"Question: {question}\n\nContext:\n{context_text}"

    # Call LLM
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0, "seed": 42}
    )
    return response['message']['content']


# Process the question
def process_question(question, qid, index, metadata, full_texts, model):
    contexts = retrieve(question, model, index, metadata)   # retrieve relevant context chunks
    raw = generate_answer(question, contexts)   # get the answer from the LLM
    print(f"\n--- RAW OUTPUT for {qid} ---\n{raw}\n---")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"answer": "I don't know", "abstained": True, "citations": []}   # Handle JSON parsing errors

    if not data.get("abstained", False):
        # validate citations
        valid_citations = []
        for cit in data.get("citations", []):
            if validate_citation(cit, full_texts):
                valid_citations.append(cit)
            else:
                print(f"  Invalid citation rejected: {cit.get('doc_id')} - {cit.get('quote', '')[:50]}...")  # Log rejected citations

        if valid_citations:
            data["citations"] = valid_citations  # keep only valid citations
            print(f"  {len(valid_citations)} citations validated.")
        else:
            print(f"  No valid citations. Abstaining.")
            data["abstained"] = True
            data["answer"] = "I don't know"
            data["citations"] = []  # if citations are invalid, change to abstained

    return {
        "id": qid,
        "answer": data.get("answer", "I don't know"),
        "abstained": data.get("abstained", True),
        "citations": data.get("citations", [])
    }


# Main function to process questions from a JSON file and write results to an output file
def main():
    import argparse
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)   # Path to the JSON file containing questions
    parser.add_argument("--out", required=True)   # Path to the output JSON file
    args = parser.parse_args()

    # Load questions from the questions JSON file
    with open(args.questions, "r") as f:
        questions = json.load(f)

    index, metadata, full_texts, model = load_index()
    results = []
    for q in questions:
        # Process each question
        print(f"\nProcessing {q['id']}...")
        result = process_question(q["question"], q["id"], index, metadata, full_texts, model)
        results.append(result)

    # Write results to the answers JSON file
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Output written to {args.out}")

if __name__ == "__main__":
    main()