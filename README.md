# RAG from scratch

A minimal retrieval-augmented generation (RAG) prototype in plain Python. It doesn't use
LangChain. Vectors are stored in a single SQLite file with [sqlite-vec](https://github.com/asg017/sqlite-vec),
and everything runs on local models through [Ollama](https://ollama.com).
Based on this video tutorial: https://youtu.be/f9uAG3AQGWM

## How it works

1. **Models** (`rag.py`): `nomic-embed-text` turns text into vectors, and a chat model writes the answers.
2. **Chunking** (`chunker.py`): every `.md` file under `DOCS_DIR` (recursively, skipping hidden folders) is split
   into one chunk per H1–H3 section. Code blocks are kept whole; one longer than the limit is split into separately fenced pieces. YAML frontmatter is removed from the text
   and kept as metadata. Sections longer than 2,000 characters are split on paragraph, table, and code-block
   boundaries, and large tables repeat their header row in every piece. Each chunk starts with its source path and
   heading trail, e.g. `handbook.md > Byte Bistro Employee Handbook > Alarm code`.
3. **Embedding** (`index.py`): chunks are embedded in batches of 64 and stored in `.rag_index/<folder>-<hash>.db`.
   On each start, only files whose content changed are re-embedded, and deleted files are removed from the index.
   Changing `EMBED_MODEL` or the chunking settings rebuilds the whole index automatically.
4. **Retrieval**: the question is embedded, and sqlite-vec returns the top-k chunks by cosine distance.
5. **Generation**: the retrieved chunks and the question go into one prompt sent to `/api/chat`, which is asked
   to cite the source files it used.

## Setup

Common tasks are recipes in the `justfile` ([just](https://just.systems)); run `just` to list them.
The equivalent raw commands are shown in comments.

Ollama runs in Docker (`Dockerfile` + `compose.yaml`). On first start the container pulls
`nomic-embed-text:v1.5` and `qwen3:4b-instruct-2507-q4_K_M` into the `ollama-models` volume, so later starts are fast.

```bash
just up        # docker compose -f compose.yaml up -d --build  (Ollama on localhost:11434)
just logs      # docker compose -f compose.yaml logs -f        (wait for "Ollama is ready")
just install   # uv sync
```

To pull a different chat model, set `OLLAMA_MODELS_TO_PULL` before starting, e.g.
`OLLAMA_MODELS_TO_PULL="nomic-embed-text:v1.5 llama3.2" just up`.
For an NVIDIA GPU, uncomment the `deploy` block in `compose.yaml`.

Stop it with `just down`. `just prune` also deletes the downloaded models (`down -v`).
`just shell` opens a shell in the Ollama container.

## Run

```bash
just run                       # uv run rag.py — interactive Q&A over docs/
just run-context               # uv run rag.py --show-context — also print the retrieved chunks
CHAT_MODEL=llama3.2 just run   # use a different chat model (pull it first, see Setup)
```

Try: "What's the alarm code?", "Can I bring my dog inside?", "Is the cafe open on Christmas?"

To use your own data, drop more `.md` files into `docs/`, or point `DOCS_DIR` at another folder:

```bash
DOCS_DIR=~/_payoy/tools/money-documentation just run   # first run embeds every file; later runs only changed ones
just index                                             # sync the index and exit, without starting the Q&A
just reindex                                           # rebuild the whole index from scratch
just clean-index                                       # delete all saved indexes
```

Each docs folder gets its own index file, so switching `DOCS_DIR` doesn't trigger a rebuild. The first full
index of `money-documentation` (109 files, 1,690 chunks) took about 5.5 minutes on CPU. A run with no changes
starts in under a second.

Environment variables: `OLLAMA_URL` (default `http://localhost:11434`), `EMBED_MODEL`, `CHAT_MODEL`,
`DOCS_DIR` (default `docs`), `INDEX_DB` (overrides the index file path), `TOP_K` (chunks per answer, default 3).
