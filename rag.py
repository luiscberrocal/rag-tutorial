"""RAG from scratch: no LangChain, just Ollama, numpy, and a SQLite vector index (sqlite-vec)."""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import requests

import index

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs")).expanduser().resolve()
TOP_K = int(os.getenv("TOP_K", "3"))
SHOW_CONTEXT = "--show-context" in sys.argv
REINDEX = "--reindex" in sys.argv
INDEX_ONLY = "--index-only" in sys.argv

# One index file per docs folder, so switching DOCS_DIR back and forth doesn't force a rebuild.
_slug = f"{DOCS_DIR.name}-{hashlib.sha256(str(DOCS_DIR).encode()).hexdigest()[:8]}"
INDEX_DB = Path(os.getenv("INDEX_DB", f".rag_index/{_slug}.db"))


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


# 2 + 3. Chunking and embedding (chunker.py, index.py): each H1-H3 section of every .md file
# under DOCS_DIR becomes a chunk. Chunks are embedded and stored in SQLite. Only files that
# changed since the last run are re-embedded.
db = index.connect(INDEX_DB)
print(f"Docs: {DOCS_DIR}\nIndex: {INDEX_DB}")
index.sync(db, DOCS_DIR, embed, EMBED_MODEL, force=REINDEX)
if INDEX_ONLY:
    sys.exit()


# 4. Retrieval: the k chunks closest to the question by cosine distance.
def retrieve(question, k=TOP_K):
    return index.search(db, embed([question])[0], k)


# 5. Augment the prompt with the retrieved chunks and generate the answer.
while True:
    try:
        question = input("\nQuestion (blank to quit): ").strip()
    except EOFError:
        break
    if not question:
        break
    hits = retrieve(question)
    if SHOW_CONTEXT:
        print("\n--- context ---")
        for _, _, text, distance in hits:
            print(f"[distance {distance:.3f}] {text}\n")
        print("---------------")
    context = "\n\n---\n\n".join(text for _, _, text, _ in hits)
    prompt = (
        "Answer the question using only the context below. Each excerpt starts with its "
        "source file and section. Cite the source file(s) you used. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    print(ask_llm(prompt))
