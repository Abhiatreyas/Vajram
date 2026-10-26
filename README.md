# Vajram

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm_NC_1.0-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Vajram--KN1--8B-yellow)](https://huggingface.co/AbhiAtreyas/Vajram-KN1-8B)
[![Demo](https://img.shields.io/badge/Live_Demo-langbrains.in-green)](https://langbrains.in)

**Vajram** (8.4B parameters) is a Kannada-first, multilingual Indic language model built on AI2's OLMo-3 architecture. It features a custom partitioned tokenizer with dedicated sub-vocabularies for Indic scripts, continued pretraining across ~5.7B tokens, and on-policy DPO.

This repository contains:
- **`tokenizer/`**: The partitioned Indic Unicode tokenizer.
- **`serve/`**: OpenAI-compatible inference server supporting token-ID streaming with `llama.cpp` and `transformers`.
- **`scripture/`**: Verified scripture collection and resolver (Bhagavad Gita and principal Upanishads) for verbatim recitation.
- **`docs/`**: Interactive evaluation charts and benchmark comparisons ([Live Report](https://abhiatreyas.github.io/vajram/)).
- **`tests/`**: Pytest test suite for tokenizer, security boundaries, and tool calling.

---

## Quickstart

Because Vajram's partitioned tokenizer uses multiple Unicode sub-models, standard `llama.cpp`, Ollama, and LM Studio cannot parse its tokens directly. The Vajram server bridges this by running tokenization in Python and sending raw token IDs directly to `llama-server`, providing a standard OpenAI-compatible API.

### 1. Download Model & Tokenizer
Request gated access at [Hugging Face: AbhiAtreyas/Vajram-KN1-8B](https://huggingface.co/AbhiAtreyas/Vajram-KN1-8B), then download:

```bash
# Using huggingface-cli
huggingface-cli download AbhiAtreyas/Vajram-KN1-8B --local-dir ./model
```

### 2. Start llama-server (Backend)
Vajram is optimized to run with `llama.cpp` using the Q8_0 GGUF (`./model/gguf/vajram-r6c-dpo-Q8_0.gguf`):

```bash
# Start llama-server on port 8090
llama-server \
  -m ./model/gguf/vajram-r6c-dpo-Q8_0.gguf \
  -c 16384 \
  -np 1 \
  --port 8090 \
  --host 127.0.0.1
```

### 3. Start the Vajram Server
Install dependencies:
```bash
pip install fastapi uvicorn tokenizers requests ddgs
```

Run the server:
```bash
export VAJRAM_MODEL_DIR="./model"
export VAJRAM_LLAMA_URL="http://127.0.0.1:8090"
export VAJRAM_PORT=8080

python serve/server.py
```

---

## Client Usage Examples

The server implements the OpenAI `/v1/chat/completions` specification.

### Using `curl` (Streaming)
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vajram-8b",
    "messages": [
      {"role": "user", "content": "ಕರ್ನಾಟಕದ ಇತಿಹಾಸದ ಬಗ್ಗೆ 3 ಮುಖ್ಯ ಅಂಶಗಳನ್ನು ತಿಳಿಸಿ."}
    ],
    "stream": true,
    "temperature": 0.3
  }'
```

### Using OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none"  # or your VAJRAM_API_KEY
)

response = client.chat.completions.create(
    model="vajram-8b",
    messages=[
        {"role": "user", "content": "ಹೇಗಿದ್ದೀರಿ? ಇವತ್ತು ಹವಾಮಾನ ಹೇಗಿದೆ ಬೆಂಗಳೂರಿನಲ್ಲಿ?"}
    ],
    temperature=0.3
)

print(response.choices[0].message.content)
```

---

## Tool Calling

Vajram supports two tool-calling modes:

### 1. Server-Side Built-in Tools
The server natively executes 7 tool schemas:
- `calculator` (AST-restricted, sandboxed math evaluator)
- `web_search` (Keyless DuckDuckGo search + page synthesis)
- `scripture_lookup` (Exact verse retrieval from verified Gita & Upanishad stores)
- `weather` (Current temperatures via wttr.in)
- `currency_convert` (Real-time ECB rates via frankfurter.app)
- `get_datetime` (Timezone-aware datetime)
- `fetch_page` (SSRF-safe web scraper)

When a prompt triggers a built-in tool, the server executes it and feeds the result back to the model automatically.

### 2. Client-Declared Custom Tools (OpenAI Format)
You can declare custom application tools in your API request. The model will output structured `tool_calls` for your application to execute:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a domestic flight ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["source", "destination", "date"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="vajram-8b",
    messages=[
        {"role": "user", "content": "Bengaluru inda Mumbai ge 28th October ge flight book maadi."}
    ],
    tools=tools
)

# Output contains assistant tool_calls
print(response.choices[0].message.tool_calls)
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `VAJRAM_BACKEND` | `llama` (recommended) or `torch` | `llama` |
| `VAJRAM_MODEL_DIR` | Path to downloaded model folder | `./model` |
| `VAJRAM_LLAMA_URL` | URL of the running llama-server instance | `http://127.0.0.1:8090` |
| `VAJRAM_PORT` | Port for the Vajram API server | `8080` |
| `VAJRAM_HOST` | Host address to bind | `0.0.0.0` |
| `VAJRAM_API_KEY` | Optional bearer token authentication | *(none)* |
| `VAJRAM_MAX_CTX` | Maximum context length | `16384` |
| `VAJRAM_TEMPERATURE` | Sampling temperature | `0.3` |
| `VAJRAM_UNSAFE_PYTHON` | Allow arbitrary Python via subprocess (not recommended) | `0` |
| `SCRIPTURE_DIR` | Path to scripture store directory | `../scripture` |

---

## Limitations

- **Indic World Knowledge:** Due to a compact continued-pretraining budget (~5.7B tokens), general Indic trivia, history, and exams (MILU 34.3%) are less comprehensive than larger 12B+ models.
- **Custom Tool Arguments:** The model may occasionally assume missing parameters instead of asking for clarification.
- **Single-Context llama.cpp Constraint:** The model requires Python-driven tokenization and cannot be directly dropped into LM Studio, vanilla llama.cpp CLI, or Ollama.

---

## License & Attribution

- **Code:** Licensed under [PolyForm Noncommercial License 1.0.0](LICENSE).
- **Weights:** Hugging Face weights are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- For third-party notices and attributions, see [NOTICE](NOTICE).
