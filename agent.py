import os
import sys
import json
import argparse
import requests
from sentence_transformers import SentenceTransformer
import faiss
import fitz  

def load_pdfs(pdf_folder):
    pdf_texts = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            try:
                doc = fitz.open(path) 
                text = ""
                for page in doc:
                    text += page.get_text()
                pdf_texts.append((filename, text))
            except Exception as e:
                print(f"Error reading {filename}: {e}", file=sys.stderr)
    return pdf_texts

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

def build_vector_index(chunks, embed_model):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1] if hasattr(embeddings, 'shape') else len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_chunks(question, embed_model, index, chunks, top_k=3):
    q_embedding = embed_model.encode([question])
    distances, indices = index.search(q_embedding, top_k)
    retrieved = []
    for idx in indices[0]:
        if idx < len(chunks):
            retrieved.append(chunks[idx])
    return retrieved

def main():
    parser = argparse.ArgumentParser(description="Customer Service Agent(OFFLINE)")
    parser.add_argument("--dir", type=str, required=True,
                        help="Path to folder containing sources")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to use (e.g., mistralai/mistral-large-2411)")
    args = parser.parse_args()
    print("Loading and extracting text from PDFs...")
    docs = load_pdfs(args.dir)
    if not docs:
        print("No PDFs found in the directory. Exiting.", file=sys.stderr)
        return
    all_chunks = []
    for filename, text in docs:
        text_chunks = chunk_text(text)
        text_chunks = [f"From {filename}: {chunk}" for chunk in text_chunks if chunk]
        all_chunks.extend(text_chunks)
    if not all_chunks:
        print("No text chunks to index. Exiting.", file=sys.stderr)
        return
    print("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding {len(all_chunks)} chunks and building FAISS index...")
    index = build_vector_index(all_chunks, embed_model)
    history_file = "conversation_history.jsonl"
    print("Setup complete. Enter your questions (type 'exit' to quit).")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        relevant = retrieve_chunks(question, embed_model, index, all_chunks, top_k=3)
        context_text = "\n\n".join(relevant)
        system_prompt = "You are a helpful assistant. Answer using only the provided context."
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        payload = {
            "model": args.model,
            "messages": messages
        }
        headers = {
            "Authorization": f"Bearer X"
        }

        print("Querying the Mistral model...")
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error from OpenRouter API: {e}", file=sys.stderr)
            continue
        print("\nAI Answer:")
        print(answer)

        try:
            with open(history_file, "a") as hf:
                record = {"question": question, "answer": answer}
                hf.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"Warning: Could not write history file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
