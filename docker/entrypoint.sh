#!/bin/sh
# Start the Ollama server, make sure the models this project needs are pulled,
# then keep the server in the foreground.
set -e

# The healthcheck passes only once this file exists, i.e. after every model is pulled.
READY_FILE=/tmp/ollama-ready
rm -f "$READY_FILE"

ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama to start..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

for model in $OLLAMA_MODELS_TO_PULL; do
  if ollama list | awk 'NR>1 {print $1}' | grep -qx "$model" \
     || ollama list | awk 'NR>1 {print $1}' | grep -qx "$model:latest"; then
    echo "Model $model already present"
  else
    echo "Pulling $model..."
    ollama pull "$model"
  fi
done

touch "$READY_FILE"
echo "Ollama is ready"
wait "$SERVER_PID"
