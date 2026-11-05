# RAG from scratch

A minimal retrieval-augmented generation (RAG) prototype in plain Python. It doesn't use
LangChain or a vector database. It runs entirely on local models through [Ollama](https://ollama.com).
Based on this video tutorial: https://youtu.be/f9uAG3AQGWM

## How it works (`rag.py`)

1. **Models**: `nomic-embed-text` turns text into vectors, and a chat model writes the answers.
2. **Chunking**: every paragraph of every `docs/*.md` file becomes one chunk.
3. **Embedding**: all chunks are embedded in one batch call to `/api/embed`.
4. **Retrieval**: the question is embedded, and the top-k chunks are picked by cosine similarity (numpy).
5. **Generation**: the retrieved chunks and the question go into one prompt sent to `/api/chat`.

## Setup

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct   # or use any chat model you already have (see below)
uv sync
```

## Run

```bash
uv run rag.py                       # interactive Q&A over docs/
uv run rag.py --show-context        # also print the retrieved chunks
CHAT_MODEL=llama3.2 uv run rag.py   # use a different chat model
```

Try: "What's the alarm code?", "Can I bring my dog inside?", "Is the cafe open on Christmas?"

To use your own data, drop more `.md` files into `docs/`. Paragraphs are separated by blank lines.

Environment variables: `OLLAMA_URL` (default `http://localhost:11434`), `EMBED_MODEL`, `CHAT_MODEL`.
