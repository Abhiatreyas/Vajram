"""Tests for the Vajram partitioned tokenizer.
"""
import os
import sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_DIR = os.path.abspath(os.path.join(HERE, "..", "tokenizer"))
HF_RELEASE_DIR = r"E:\LLM_exp\release\hf"

sys.path.insert(0, TOKENIZER_DIR)
import vajram_tokenizer as vt


@pytest.fixture(scope="module")
def tokenizer():
    # Load from the HF release snapshot (single source of truth for vocab files)
    vocab_dir = HF_RELEASE_DIR if os.path.exists(os.path.join(HF_RELEASE_DIR, "vajram_layout.json")) else TOKENIZER_DIR
    return vt.VajramTokenizer.from_dir(vocab_dir)


def test_roundtrip_english(tokenizer):
    text = "The quick brown fox jumps over the lazy dog. 1234567890! @#$%^&*()"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_kannada(tokenizer):
    text = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು. ಕನ್ನಡ ಭಾಷೆ ಅತ್ಯಂತ ಪ್ರಾಚೀನ ಮತ್ತು ಸುಂದರವಾಗಿದೆ."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_hindi(tokenizer):
    text = "नमस्ते, भारत एक विशाल और सुंदर देश है। यहाँ कई भाषाएँ बोली जाती हैं।"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_kanglish(tokenizer):
    text = "naanu naale barthini, swalpa wait maadi. hegiddira sir?"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_code(tokenizer):
    text = "def calculate_factorial(n: int) -> int:\n    return 1 if n <= 1 else n * calculate_factorial(n - 1)\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_math(tokenizer):
    text = "E = mc^2; \\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}; 15% of 800 = 120"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_fallback(tokenizer):
    # Russian (Cyrillic) and Chinese (CJK) fall back to OLMo's byte tokens
    text = "Привет мир! 你好世界!"
    encoded = tokenizer.encode(text)
    assert len(encoded) > 0
    assert tokenizer.decode(encoded) == text


def test_control_tokens(tokenizer):
    # Control tokens are non-special tokens in the layout
    controls = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<|act_direct|>", "<|act_tool|>", "<|lang_kn|>", "<|lang_en|>"]
    for c in controls:
        ids = tokenizer.encode(c)
        assert len(ids) == 1, f"Expected 1 token for control token {c}, got {ids}"
        assert tokenizer.decode(ids) == c


def test_kannada_density(tokenizer):
    # Verify Kannada token density is high (few tokens per sentence)
    text = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು."
    ids = tokenizer.encode(text)
    assert len(ids) <= 5, f"Expected <= 5 tokens for sentence, got {len(ids)}"
