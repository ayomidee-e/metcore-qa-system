import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import INDEX_DIR, EMBEDDING_MODEL, TOP_K

# Load FAISS index and metadata
def load_index_and_metadata():
    index = faiss.read_index(f"{INDEX_DIR}/faiss_index.bin")
    with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer(EMBEDDING_MODEL)  # load sentence transformer model
    return index, metadata, model   # return all loaded components

# Search for relevant documents
def search(query, model, index, metadata, top_k=TOP_K):
    emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(emb.astype(np.float32), top_k)    # search the FAISS index for the most similar chunks
    results = []
    for i, idx in enumerate(indices[0]):    # loop through the retrieved indices
        if idx != -1:   # check if the index is valid
            results.append({
                "score": float(distances[0][i]),    # store the similarity score
                "chunk": metadata[idx]  # store the corresponding chunk metadata
            })
    return results  # return all retrieved chunks

# Main function to run the search interface on user queries
def main():
    index, metadata, model = load_index_and_metadata()  # load all components
    print("Retrieval ready. Type 'exit' to quit.")
    while True:
        q = input("\n Query: ").strip()  # get the user query
        if q.lower() == "exit":
            break
        results = search(q, model, index, metadata)  # search for relevant documents
        print(f"\nTop {len(results)} chunks:")
        for i, r in enumerate(results, 1):  # loop through the retrieved chunks
            chunk = r["chunk"]
            print(f"\n--- #{i} (score={r['score']:.4f}) ---")
            print(f"Doc: {chunk['doc_id']}  |  Page: {chunk['page']}  |  ID: {chunk['chunk_id']}")
            print(f"Text: {chunk['text'][:300]}...")

if __name__ == "__main__":
    main()