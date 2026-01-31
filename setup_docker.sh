#!/bin/bash

if ! command -v docker &> /dev/null; then
    echo "docker not installed"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "docker compose not installed"
    exit 1
fi

docker compose build
docker compose up -d

sleep 5

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    if ! ollama list | grep -q "phi4-mini:3.8b"; then
        echo "pulling phi4-mini:3.8b..."
        ollama pull phi4-mini:3.8b
    fi
else
    echo "ollama not running at localhost:11434"
    echo "run: ollama serve"
    echo "then: ollama pull phi4-mini:3.8b"
fi
