# Vajram Tokenizer

The **Vajram Partitioned Tokenizer** is a custom Unicode-routed tokenizer designed for high-density tokenization of Indic languages without compromising English, code, or mathematics.

## Architecture

Standard language models apply a single BPE or WordPiece model across all scripts. When trained with low-resource Indic languages, this produces severe fragmentation (often requiring 4× to 6× more tokens per word than English) and script bleed.

Vajram uses a deterministic, Unicode-based routing mechanism:
1. **Base OLMo-3 Range `[0, 100278)`:** Retains AI2's OLMo-3 byte-level BPE vocab and merges exactly. English, programming languages, mathematical expressions, punctuation, and ASCII whitespace are tokenized identically to OLMo-3.
2. **Dedicated Indic Slots `[CONTROL_END, ...)`:** Contiguous runs of Indic scripts are routed directly to dedicated Unigram sub-vocabularies:
   - `KANNADA`
   - `DEVANAGARI` (Hindi, Marathi, Sanskrit)
   - `TELUGU`
   - `TAMIL`
   - `MALAYALAM`
   - `BENGALI`
   - `GUJ_PA` (Gujarati, Punjabi)
   Spaces preceding an Indic word are owned by the Indic run (leading-whitespace ownership). ZWJ, ZWNJ, and danda characters are preserved within the script block.
3. **Romanized Indic Merges `[100278, CONTROL_START)`:** Syllables and prefixes common to Kanglish and Hinglish.
4. **Control Tokens `[CONTROL_START, CONTROL_END)`:** Language tags (`<|lang_kn|>`, `<|lang_hi|>`, etc.), action tags (`<|act_direct|>`, `<|act_tool|>`), and tool tokens (`<tool_call>`, `</tool_call>`).
5. **Byte Fallback (Catchall):** Scripts without an allocated slot (e.g. Cyrillic, CJK, Arabic) fall back cleanly to OLMo-3's 256 byte tokens.

---

## Measured Token Density (FLORES+ dev)

Characters per token, measured on FLORES+ development text:

| Language | Vajram | Qwen3-8B | Gemma-3 12B | Sarvam-M | Krutrim-2 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Kannada** | **4.8** | 0.7 | 2.6 | 2.2 | 2.2 |
| **Hindi** | **4.5** | 1.1 | 3.7 | 2.6 | 2.6 |
| **Telugu** | **4.2** | 0.7 | 2.8 | 2.1 | 2.1 |
| **English** | 4.85 | 4.8 | 4.8 | 4.7 | 4.7 |

---

## Vocabulary Files

Vocabularies are distributed alongside model weights in the Hugging Face repository snapshot:
`https://huggingface.co/AbhiAtreyas/Vajram-KN1-8B`

- `vajram_layout.json`: Slot offsets, control token mappings, and total vocab size (236,544).
- `olmo_slot.json`: Base BPE model.
- `slots/*.json`: Unigram tokenizer models for each Indic script.

To use the tokenizer:
```python
from transformers import AutoTokenizer

# The Hugging Face repo acts as the single source of truth for vocab files
tok = AutoTokenizer.from_pretrained("AbhiAtreyas/Vajram-KN1-8B", trust_remote_code=True)
ids = tok.encode("ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು.")
print("Tokens:", ids)
print("Decoded:", tok.decode(ids))
```

---

## Compatibility Note

Because of the multi-slot partitioned structure, **the tokenizer cannot be serialized into a single standard `tokenizer.json` file**. As a result:
- It runs seamlessly in Python via `tokenizers` and `transformers` (`trust_remote_code=True`).
- Direct loading in vanilla `llama.cpp`, Ollama, or LM Studio is **not supported**. For local inference, use the inference server in `../serve/`.

## License
PolyForm Noncommercial License 1.0.0.
