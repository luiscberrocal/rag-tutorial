set dotenv-load
set shell := ["bash", "-euo", "pipefail", "-c"]

# Overrides any COMPOSE_FILE set in the shell so every recipe uses this project's compose file
export COMPOSE_FILE := "compose.yaml"

container_name := "rag-ollama"

# List all available commands
default:
    @just --list

# Build the Ollama image
build *args:
    @docker compose build {{args}}

# Build if needed and start Ollama in the background (pulls missing models on first start)
up:
    @docker compose up -d --build

# Stop containers
down:
    @docker compose down

# Stop containers and delete the model volume (destructive — models must be downloaded again)
prune:
    @docker compose down -v

# Tail container logs; wait for "Ollama is ready" after the first start
logs *args:
    @docker compose logs -f {{args}}

# Open a shell inside the running Ollama container
shell:
    @docker compose exec ollama bash

# Install Python dependencies with uv
install:
    @uv sync

# Run the interactive Q&A over docs/ (e.g. `just run --show-context`)
run *args:
    @uv run rag.py {{args}}

# Run the Q&A and also print the retrieved chunks
run-context:
    @uv run rag.py --show-context

# Sync the index (re-embeds only changed files) and exit; set DOCS_DIR to index another folder
index:
    @uv run rag.py --index-only

# Rebuild the whole index from scratch, then exit
reindex:
    @uv run rag.py --index-only --reindex

# Delete every saved index in .rag_index/ (each docs folder is fully re-embedded on its next run)
clean-index:
    @rm -rf .rag_index

# Remove Python cache artifacts
clean:
    @find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +

# Remove a stale Ollama container left from a previous run
clean-containers:
    @docker rm -f {{container_name}} 2>/dev/null || true
