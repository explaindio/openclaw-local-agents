#!/bin/bash
# Needle In A Haystack (NIAH) test for llama-server
# Usage: ./niah_test.sh <context_size>
# Requires: llama-server running on localhost:8080

set -e

CONTEXT_SIZE=${1:-65536}
SERVER_URL="http://127.0.0.1:8080/v1/chat/completions"
NEEDLE="The secret launch code is PINEAPPLE-774."
QUESTION="What is the secret launch code? Reply with ONLY the launch code, nothing else."

# Reserve ~500 tokens for needle + question + response
FILLER_TOKENS=$((CONTEXT_SIZE - 500))

echo "=== Needle In A Haystack Test ==="
echo "Target context: ${CONTEXT_SIZE} tokens"
echo "Filler tokens: ~${FILLER_TOKENS}"
echo ""

# Generate filler text (~4 chars per token as a rough estimate)
FILLER_CHARS=$((FILLER_TOKENS * 4))

# Create filler: repeated paragraph about various mundane topics
FILLER_PARAGRAPH="The history of papermaking dates back to ancient China, where Cai Lun is credited with inventing the process around 105 AD. The technique spread along the Silk Road to the Islamic world and eventually to Europe. Early paper was made from bark, hemp, and rags. Modern papermaking uses wood pulp and involves complex chemical processes. The industry has evolved significantly with the advent of digital technology, though paper remains an essential material in many applications. Environmental concerns have led to increased recycling efforts and the development of sustainable forestry practices. The papermaking process involves several steps: pulping, bleaching, forming, pressing, and drying. Different types of paper are produced for various purposes, including writing, printing, packaging, and specialty applications. "

# Build the filler text by repeating the paragraph
echo "Generating filler text (~${FILLER_CHARS} chars)..."
FILLER=""
while [ ${#FILLER} -lt $FILLER_CHARS ]; do
    FILLER="${FILLER}${FILLER_PARAGRAPH}"
done
FILLER="${FILLER:0:$FILLER_CHARS}"

echo "Filler text generated: ${#FILLER} chars"

# Construct the prompt: NEEDLE at the beginning, filler in the middle, question at the end
PROMPT="${NEEDLE}\n\n${FILLER}\n\n${QUESTION}"

# Escape for JSON
PROMPT_JSON=$(python3 -c "
import json, sys
needle = '''${NEEDLE}'''
filler_file = '/tmp/niah_filler.txt'
question = '''${QUESTION}'''

# Read filler from file
with open(filler_file, 'r') as f:
    filler = f.read()

prompt = needle + '\n\n' + filler + '\n\n' + question
msg = {'role': 'user', 'content': prompt}
print(json.dumps(msg))
")

# Save filler to temp file first
echo "$FILLER" > /tmp/niah_filler.txt

# Build JSON payload
PAYLOAD=$(python3 -c "
import json

needle = '${NEEDLE}'
question = '${QUESTION}'

with open('/tmp/niah_filler.txt', 'r') as f:
    filler = f.read()

prompt = needle + '\n\n' + filler + '\n\n' + question

payload = {
    'model': 'qwen3.5',
    'messages': [
        {'role': 'user', 'content': prompt}
    ],
    'max_tokens': 100,
    'temperature': 0.1
}

print(json.dumps(payload))
")

echo ""
echo "Sending request to llama-server..."
echo "Prompt size: $(echo "$PAYLOAD" | wc -c) bytes"
echo ""

# Log VRAM before
echo "=== VRAM BEFORE ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Send the request
RESPONSE=$(curl -s -X POST "$SERVER_URL" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    --max-time 300)

# Log VRAM after
echo ""
echo "=== VRAM AFTER ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo ""
echo "=== RESPONSE ==="
echo "$RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    content = data['choices'][0]['message']['content']
    print(f'Model response: {content}')
    if 'PINEAPPLE-774' in content:
        print('✅ NEEDLE FOUND — Model comprehends at this context length!')
    else:
        print('❌ NEEDLE NOT FOUND — Model failed to retrieve the information.')
    
    # Print usage stats if available
    if 'usage' in data:
        usage = data['usage']
        print(f\"Tokens — prompt: {usage.get('prompt_tokens', 'N/A')}, completion: {usage.get('completion_tokens', 'N/A')}, total: {usage.get('total_tokens', 'N/A')}\")
except Exception as e:
    print(f'Error parsing response: {e}')
    print(f'Raw response: {sys.stdin.read()[:500]}')
"

echo ""
echo "=== PEAK VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader
echo ""
echo "=== TEST COMPLETE ==="
