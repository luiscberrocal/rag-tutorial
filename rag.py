"""RAG from scratch: no LangChain, no vector database, just Ollama + numpy."""
import os
import sys
from pathlib import Path

import numpy as np
import requests

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
SHOW_CONTEXT = "--show-context" in sys.argv


# 1. Models: one turns text into vectors, the other answers questions.
def ollama(path, payload):
    r = requests.post(f"{OLLAMA}{path}", json=payload)
    if not r.ok:  # surface Ollama's own message, e.g. "model '...' not found"
        sys.exit(f"Ollama {path} failed ({r.status_code}): {r.text.strip()}")
    return r.json()


def embed(texts):
    return np.array(ollama("/api/embed", {"model": EMBED_MODEL, "input": texts})["embeddings"])


def ask_llm(prompt):
    return ollama("/api/chat", {
        "model": CHAT_MODEL, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    })["message"]["content"]


# 2. Chunking: every non-empty paragraph of every docs/*.md file is one chunk.
chunks = []
for file in sorted(Path("docs").glob("*.md")):
    chunks += [p.strip() for p in file.read_text().split("\n\n") if p.strip()]

# 3. Embedding: one vector per chunk, computed in a single batch request.
chunk_vectors = embed(chunks)
print(f"Indexed {len(chunks)} chunks")


# 4. Retrieval: cosine similarity between the question and every chunk.
def retrieve(question, k=3):
    q = embed([question])[0]
    scores = chunk_vectors @ q / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q))
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]


# 5. Augment the prompt with the retrieved chunks and generate the answer.
while True:
    question = input("\nQuestion (blank to quit): ").strip()
    if not question:
        break
    context = "\n\n".join(retrieve(question))
    if SHOW_CONTEXT:
        print(f"\n--- context ---\n{context}\n---------------")
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    print(ask_llm(prompt))
