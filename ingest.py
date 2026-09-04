import os
import pickle
import numpy as np
import pdfplumber
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from config import PDF_DIR, INDEX_DIR, CHUNK_SIZE, OVERLAP, EMBEDDING_MODEL

# Extract text from PDF
def extract_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):   # Loop through each page (starting page count at 1)
            text = page.extract_text()  # Extract text from that page
            if text and text.strip():   # Check if text exists and isn't just empty spaces
                pages.append((i, text.strip()))  # Save the page number and its text as a pair
    return pages


# Chunk text into smaller segments
def chunk_text(text, page_num, doc_id, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    text_len = len(text)  # Get the length of the text
    chunk_counter = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)  # Get the end index for the chunk
        if end < text_len:
            # Trying to cut at a sentence boundary (., !, or ?) if possible (last 50 characters)
            for j in range(min(end, text_len-1), max(start, end-50), -1):
                if text[j] in '.!?':
                    end = j + 1
                    break
        # If no sentence boundary is found, cut at the original end
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                # Create a dictionary:
                "doc_id": doc_id,  # document ID
                "page": page_num,  # page number
                "text": chunk_text,  # chunk text
                "chunk_id": f"{doc_id}_p{page_num}_c{chunk_counter}"  # unique ID for this chunk
            })
            chunk_counter += 1  # increment chunk counter but keep some overlap from the previous chunk (so context doesn't get lost)
        start = end - overlap if end < text_len else text_len  # continue until all text is chunked
    return chunks   # Return the list of chunks


# Main ingestion function
def main():
    print("1. Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)  # Load the embedding model (this converts text to numbers)

    # Load PDF files
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{PDF_DIR}'. Please add the four documents.")
        return

    print(f"2. Found {len(pdf_files)} PDFs. Processing...")
    # Create an empty list to store chunks and dictionary for full texts
    all_chunks = []
    full_texts = {}

    # Extract text from PDFs
    for pdf_file in tqdm(pdf_files):
        doc_id = os.path.splitext(pdf_file)[0]
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        pages = extract_pages(pdf_path)

        # Combine all page texts into one full document
        full_text = "\n".join([text for _, text in pages])
        full_texts[doc_id] = full_text

        # Chunk text into smaller segments
        for page_num, text in pages:
            chunks = chunk_text(text, page_num, doc_id)
            all_chunks.extend(chunks)

    print(f"3. Created {len(all_chunks)} total chunks.")

    # Embedding
    texts = [c["text"] for c in all_chunks]
    print("4. Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # using Euclidean distance
    index.add(embeddings.astype(np.float32))

    # Save index
    faiss_path = os.path.join(INDEX_DIR, "faiss_index.bin")
    faiss.write_index(index, faiss_path)
    print(f"5. Saved FAISS index to {faiss_path}")

    # Save metadata
    meta_path = os.path.join(INDEX_DIR, "metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"6. Saved metadata to {meta_path}")

    # Save full_texts
    full_texts_path = os.path.join(INDEX_DIR, "full_texts.pkl")
    with open(full_texts_path, "wb") as f:
        pickle.dump(full_texts, f)
    print(f"7. Saved full_texts to {full_texts_path}")

    print("Ingestion complete!")

if __name__ == "__main__":
    main()