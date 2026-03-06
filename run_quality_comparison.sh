#!/bin/bash
# Run KV quality comparison for all 4 configurations
set -e

LLAMA_SERVER="/home/code10/qwen3_5/llama.cpp/build/bin/llama-server"
export LD_LIBRARY_PATH="/home/code10/qwen3_5/llama.cpp/build/bin:$LD_LIBRARY_PATH"
RESULTS_DIR="/home/code10/qwen3_5/quality_results"
mkdir -p "$RESULTS_DIR"

start_server() {
    local model=$1 ctx=$2 kv_mode=$3 label=$4
    echo ""
    echo "================================================================"
    echo "  Starting: $label"
    echo "  Model: $(basename $model), Context: $ctx, KV: $kv_mode"
    echo "================================================================"
    
    pkill -f llama-server 2>/dev/null || true
    sleep 3
    
    if [ "$kv_mode" = "q8_0" ]; then
        nohup $LLAMA_SERVER -m "$model" -c "$ctx" -ngl 99 \
            -ctk q8_0 -ctv q8_0 --flash-attn on \
            --host 127.0.0.1 --port 8080 > /tmp/llama_quality.log 2>&1 &
    else
        nohup $LLAMA_SERVER -m "$model" -c "$ctx" -ngl 99 \
            --flash-attn on \
            --host 127.0.0.1 --port 8080 > /tmp/llama_quality.log 2>&1 &
    fi
    
    echo "Waiting for server to be ready..."
    for i in $(seq 1 60); do
        if curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok; then
            echo "Server ready!"
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server failed to start"
    return 1
}

# Test 1: 27B f16
start_server "/home/code10/qwen3_5/models/Qwen3.5-27B-Q4_K_M.gguf" 32768 "f16" "27B-f16"
python3 /home/code10/qwen3_5/kv_quality_test.py "27B-f16" "$RESULTS_DIR/27b_f16.json"

# Test 2: 27B q8_0
start_server "/home/code10/qwen3_5/models/Qwen3.5-27B-Q4_K_M.gguf" 32768 "q8_0" "27B-q8_0"
python3 /home/code10/qwen3_5/kv_quality_test.py "27B-q8_0" "$RESULTS_DIR/27b_q8.json"

# Test 3: 35B f16
start_server "/home/code10/qwen3_5/models/Qwen3.5-35B-A3B-Q4_K_M.gguf" 32768 "f16" "35B-A3B-f16"
python3 /home/code10/qwen3_5/kv_quality_test.py "35B-A3B-f16" "$RESULTS_DIR/35b_f16.json"

# Test 4: 35B q8_0
start_server "/home/code10/qwen3_5/models/Qwen3.5-35B-A3B-Q4_K_M.gguf" 32768 "q8_0" "35B-A3B-q8_0"
python3 /home/code10/qwen3_5/kv_quality_test.py "35B-A3B-q8_0" "$RESULTS_DIR/35b_q8.json"

# Cleanup
pkill -f llama-server 2>/dev/null || true

# Compare results
echo ""
echo "================================================================"
echo "  COMPARISON SUMMARY"
echo "================================================================"
python3 -c "
import json, os

configs = [
    ('27B f16', '27b_f16.json'),
    ('27B q8_0', '27b_q8.json'),
    ('35B-A3B f16', '35b_f16.json'),
    ('35B-A3B q8_0', '35b_q8.json'),
]

dir = '$RESULTS_DIR'
all_results = {}
for label, fname in configs:
    path = os.path.join(dir, fname)
    if os.path.exists(path):
        with open(path) as f:
            all_results[label] = json.load(f)

# Print comparison table
test_names = [r['name'] for r in list(all_results.values())[0]]
print(f'{'Test':<20}', end='')
for label in all_results:
    print(f'{label:<15}', end='')
print()
print('-' * (20 + 15 * len(all_results)))

for i, name in enumerate(test_names):
    print(f'{name:<20}', end='')
    for label, results in all_results.items():
        r = results[i]
        status = '✅' if r['passed'] else '❌'
        print(f'{status} {r[\"time\"]:.1f}s{\"\":<6}', end='')
    print()

print()
for label, results in all_results.items():
    passed = sum(1 for r in results if r['passed'])
    total_time = sum(r['time'] for r in results)
    print(f'{label}: {passed}/{len(results)} passed, total time: {total_time:.1f}s')
"
echo ""
echo "Done! Results saved in $RESULTS_DIR/"
