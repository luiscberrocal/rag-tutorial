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

Ollama runs in Docker (`Dockerfile` + `compose.yaml`). On first start the container pulls
`nomic-embed-text:v1.5` and `qwen3:4b-instruct-2507-q4_K_M` into the `ollama-models` volume, so later starts are fast.

```bash
docker compose -f compose.yaml up -d --build   # start Ollama on localhost:11434
docker compose -f compose.yaml logs -f ollama  # watch the model pull; wait for "Ollama is ready"
uv sync
```

To pull a different chat model, set `OLLAMA_MODELS_TO_PULL` before starting, e.g.
`OLLAMA_MODELS_TO_PULL="nomic-embed-text:v1.5 llama3.2" docker compose -f compose.yaml up -d`.
For an NVIDIA GPU, uncomment the `deploy` block in `compose.yaml`.

Stop it with `docker compose -f compose.yaml down` (add `-v` to delete the downloaded models).

## Run

```bash
uv run rag.py                       # interactive Q&A over docs/
uv run rag.py --show-context        # also print the retrieved chunks
CHAT_MODEL=llama3.2 uv run rag.py   # use a different chat model
```

Try: "What's the alarm code?", "Can I bring my dog inside?", "Is the cafe open on Christmas?"

To use your own data, drop more `.md` files into `docs/`. Paragraphs are separated by blank lines.

Environment variables: `OLLAMA_URL` (default `http://localhost:11434`), `EMBED_MODEL`, `CHAT_MODEL`.
