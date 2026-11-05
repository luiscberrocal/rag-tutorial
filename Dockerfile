# Ollama server that pulls the models rag.py needs on first start.
# Pinned to the 0.40.14 release; bump deliberately, never use :latest.
FROM ollama/ollama:0.40.14

# Space-separated list of models pulled at startup (skipped if already in the volume).
ENV OLLAMA_MODELS_TO_PULL="nomic-embed-text qwen3:4b-instruct"
# Listen on all interfaces so the port can be published to the host.
ENV OLLAMA_HOST=0.0.0.0:11434

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 11434
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
