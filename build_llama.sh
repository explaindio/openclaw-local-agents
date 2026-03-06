#!/bin/bash
set -e

echo "=== Step 1: Building llama.cpp with CUDA ==="
cd /home/code10/qwen3_5/llama.cpp

echo "--- Running cmake ---"
cmake -B build -DGGML_CUDA=ON 2>&1

echo "--- Running cmake --build (this takes ~5-10 min) ---"
cmake --build build --config Release -j$(nproc) 2>&1 | tail -20

echo ""
echo "=== Step 2: Verify binary exists ==="
if [ -f "./build/bin/llama-server" ]; then
    echo "✅ llama-server built successfully at: ./build/bin/llama-server"
    ./build/bin/llama-server --version 2>&1 || true
else
    echo "❌ llama-server binary not found!"
    find ./build -name "llama-server" -type f 2>/dev/null
    exit 1
fi

echo ""
echo "=== BUILD COMPLETE ==="
