#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Creating Solver model..."
ollama create solver -f "$SCRIPT_DIR/Modelfile.solver"

echo "Creating Proposer model..."
ollama create proposer -f "$SCRIPT_DIR/Modelfile.proposer"

echo ""
echo "Done. Registered models:"
ollama list | grep -E "^(NAME|solver|proposer)"
