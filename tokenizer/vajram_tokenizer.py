"""
Vajram partitioned tokenizer (spec v3 §3).

Id layout
    [0, 100278)                 OLMO: OLMo 3's byte-level BPE, ids unchanged (English, code, math,
                                digits, punctuation, whitespace, Greek, accented Latin).
    [100278, CONTROL_START)     romanized-Indic BPE tokens (merges appended after OLMo's, lowest priority).
    [CONTROL_START, CONTROL_END) chat/think/tool/router/vision tokens (added as NON-special tokens so every
                                inference engine renders them in output text). Ids come from the layout.
    [CONTROL_END, ...)          Indic Unigram slots, in SLOT_ORDER.
    CATCHALL                    no rows of its own: unclaimed scripts (CJK, Cyrillic, Arabic, Odia ...)
                                are encoded as OLMo's 256 byte-level tokens.

Routing is deterministic (Unicode blocks, no language ID):
  * a maximal run of one Indic script -> that script's slot; spaces directly before it join it
    (leading-whitespace ownership); ZWJ/ZWNJ and danda stay inside the run;
  * a run of an unclaimed script -> CATCHALL bytes;
  * everything else -> OLMO, tokenized exactly as OLMo 3 does (same regex, same merges).

Only depends on `tokenizers`, so the same file runs in training data prep, HF transformers
(trust_remote_code) and eval.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from tokenizers import Tokenizer

TOKENIZER_VERSION = "vajram-v3.0"
OLMO_VOCAB = 100278

# Order = id order of the slots after CONTROL. Ranges are Unicode blocks.
INDIC_SLOTS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "KANNADA": ((0x0C80, 0x0CFF),),
    "DEVANAGARI": ((0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x1CD0, 0x1CFF)),
    "TELUGU": ((0x0C00, 0x0C7F),),
    "TAMIL": ((0x0B80, 0x0BFF),),
    "MALAYALAM": ((0x0D00, 0x0D7F),),
    "BENGALI": ((0x0980, 0x09FF),),
    "GUJ_PA": ((0x0A80, 0x0AFF), (0x0A00, 0x0A7F)),
}
SLOT_ORDER = list(INDIC_SLOTS)

# Scripts not modelled -> CATCHALL (OLMo byte tokens). Greek stays in OLMO (math symbols).
CATCHALL_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x0400, 0x052F), (0x2DE0, 0x2DFF),                      # Cyrillic
    (0x0530, 0x058F), (0x0590, 0x05FF),                      # Armenian, Hebrew
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),    # Arabic
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFE),                      # U+FEFF is the BOM, not Arabic
    (0x0B00, 0x0B7F),                                        # Odia (no corpus)
    (0x0D80, 0x0DFF), (0x0E00, 0x0EFF), (0x0F00, 0x0FFF),    # Sinhala, Thai, Lao, Tibetan
    (0x1000, 0x109F), (0x10A0, 0x10FF), (0x1100, 0x11FF),    # Myanmar, Georgian, Hangul Jamo
    (0x1200, 0x137F), (0x1780, 0x17FF),                      # Ethiopic, Khmer
    (0x3040, 0x30FF), (0x3130, 0x318F), (0x3400, 0x4DBF),    # Kana, Hangul compat, CJK ext A
    (0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0xF900, 0xFAFF),    # CJK, Hangul, CJK compat
)

ZWNJ, ZWJ = "‌", "‍"
DANDA = "।॥"

# Control tokens (spec v3 §3.4). Order is frozen once a checkpoint depends on it.
CONTROL_TOKENS: List[str] = (
    ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]
    + ["<|act_direct|>", "<|act_think|>", "<|act_tool|>", "<|act_refuse|>", "<|act_clarify|>"]
    + ["<|lang_en|>"]
    + [f"<|lang_{l}{s}|>" for l in ("kn", "hi", "te", "ta", "ml", "mr", "bn") for s in ("", "_rom")]
    + ["<|image_start|>", "<|image|>", "<|image_end|>"]
    # v3.1 (2026-09-30): typed-decision labels (SFT spec §6.1, §6.7) take the first 68 former <|reserved_N|> rows.
    # Only the strings change; ids 108307-108374 and the CPT weights are untouched.
    + ["<|decide|>", "<|yes|>", "<|no|>", "<|unknown|>"]
    + [f"<|opt_{i:02d}|>" for i in range(64)]
)
CONTROL_RESERVED = 128  # rows reserved for control tokens; unused ones are <|reserved_N|>


def _cls(ranges) -> str:
    return "".join(f"\\u{lo:04x}-\\u{hi:04x}" for lo, hi in ranges)


def _build_router() -> re.Pattern:
    parts = []
    for name, ranges in INDIC_SLOTS.items():
        c = _cls(ranges)
        # starts with a real script char; continues with joiners/danda and across spaces that are
        # followed by the same script (phrase-level runs, as measured best in E:\Tokenizer SPEC §5.1)
        body = f"[{c}][{c}{ZWNJ}{ZWJ}{DANDA}]*"
        parts.append(f"(?P<{name}> *{body}(?: +{body})*)")
    parts.append(f"(?P<CATCHALL>[{_cls(CATCHALL_RANGES)}]+)")
    return re.compile("|".join(parts))


_ROUTER = _build_router()


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2/OLMo byte-level alphabet: byte value -> printable char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def route(text: str) -> List[Tuple[str, str]]:
    """Split NFC text into ordered (slot, run) pairs. Concatenating runs reproduces the text."""
    out: List[Tuple[str, str]] = []
    pos = 0
    for m in _ROUTER.finditer(text):
        if m.start() > pos:
            out.append(("OLMO", text[pos:m.start()]))
        out.append((m.lastgroup, m.group(0)))
        pos = m.end()
    if pos < len(text):
        out.append(("OLMO", text[pos:]))
    return out


class VajramTokenizer:
    def __init__(self, olmo: Tokenizer, slots: Dict[str, Tokenizer], layout: dict):
        self.olmo = olmo
        self.slots = slots
        self.layout = layout
        self.control_end = layout["control_end"]
        self.vocab_size = layout["vocab_size"]
        self.slot_start = {s["name"]: s["start"] for s in layout["slots"]}
        self.slot_end = {s["name"]: s["end"] for s in layout["slots"]}
        # hf id -> local content id for each Indic slot (the slot's <unk> is excluded)
        self._hf2local: Dict[str, Dict[int, int]] = {}
        self._local2hf: Dict[str, List[int]] = {}
        for name, tok in slots.items():
            specials = set(tok.get_added_tokens_decoder().keys())
            ordered = sorted((i for i in tok.get_vocab().values() if i not in specials))
            self._local2hf[name] = ordered
            self._hf2local[name] = {h: l for l, h in enumerate(ordered)}
        b2u = _bytes_to_unicode()
        ov = olmo.get_vocab()
        self.byte_ids = [ov[b2u[b]] for b in range(256)]
        # id -> slot name, for decoding (flat list, O(1))
        self._id_slot: List[str] = ["OLMO"] * self.control_end
        for name in SLOT_ORDER:
            if name in self.slot_start:
                self._id_slot += [name] * (self.slot_end[name] - self.slot_start[name])
        self._ranges = [("OLMO", 0, self.control_end)] + [
            (n, self.slot_start[n], self.slot_end[n]) for n in SLOT_ORDER if n in self.slot_start]
        self.unreachable_ids = layout.get("unreachable_olmo_ids", [])

    # ---------------------------------------------------------------- io
    @classmethod
    def from_dir(cls, d: str) -> "VajramTokenizer":
        layout = json.load(open(os.path.join(d, "vajram_layout.json"), encoding="utf-8"))
        olmo = Tokenizer.from_file(os.path.join(d, "olmo_slot.json"))
        slots = {s["name"]: Tokenizer.from_file(os.path.join(d, "slots", f"{s['name']}.json"))
                 for s in layout["slots"]}
        return cls(olmo, slots, layout)

    # ---------------------------------------------------------- encoding
    def _encode_indic(self, name: str, run: str) -> List[int]:
        """Encode one Indic run; characters the slot can't represent fall back to bytes."""
        h2l, start = self._hf2local[name], self.slot_start[name]
        ids = self.slots[name].encode(run).ids
        out = []
        for h in ids:
            l = h2l.get(h)
            if l is None:
                break
            out.append(start + l)
        else:
            return out
        # Bisect so a single unrepresentable char only costs its own bytes
        if len(run) == 1:
            return [self.byte_ids[b] for b in run.encode("utf-8")]
        mid = len(run) // 2
        return self._encode_indic(name, run[:mid]) + self._encode_indic(name, run[mid:])

    def encode(self, text: str) -> List[int]:
        text = unicodedata.normalize("NFC", text)
        runs = route(text)
        # One Rust call per slot per document (per-run calls cost more than the tokenization itself)
        by_slot: Dict[str, List[str]] = {}
        for slot, run in runs:
            if slot == "OLMO" or slot in self.slots:
                by_slot.setdefault(slot, []).append(run)
        encoded = {}
        for slot, texts in by_slot.items():
            tk = self.olmo if slot == "OLMO" else self.slots[slot]
            encoded[slot] = iter(tk.encode_batch(texts, add_special_tokens=False))
        ids: List[int] = []
        for slot, run in runs:
            if slot == "OLMO":
                ids.extend(next(encoded["OLMO"]).ids)
            elif slot in self.slots:
                hf_ids = next(encoded[slot]).ids
                h2l, start = self._hf2local[slot], self.slot_start[slot]
                local = [h2l.get(h) for h in hf_ids]
                if None in local:  # slot can't represent some char: bisect, bytes for just that char
                    ids.extend(self._encode_indic(slot, run))
                else:
                    ids.extend(start + l for l in local)
            else:  # CATCHALL or a declared-but-untrained slot: bytes
                ids.extend(self.byte_ids[b] for b in run.encode("utf-8"))
        return ids

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    # ---------------------------------------------------------- decoding
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        parts: List[str] = []
        i, n = 0, len(ids)
        while i < n:
            if not 0 <= ids[i] < self.vocab_size:
                raise ValueError(f"id {ids[i]} out of range")
            slot = self._id_slot[ids[i]]
            j = i + 1
            while j < n and 0 <= ids[j] < self.vocab_size and self._id_slot[ids[j]] == slot:
                j += 1
            group = list(ids[i:j])
            if slot == "OLMO":
                parts.append(self.olmo.decode(group, skip_special_tokens=skip_special_tokens))
            else:
                start, l2h = self.slot_start[slot], self._local2hf[slot]
                parts.append(self.slots[slot].decode([l2h[t - start] for t in group], skip_special_tokens=False))
            i = j
        return "".join(parts)

    # ------------------------------------------------------- introspection
    def id_to_piece(self, i: int) -> str:
        """Human-readable piece; Indic pieces are prefixed with their slot to keep strings unique."""
        slot = self._id_slot[i]
        if slot == "OLMO":
            return self.olmo.id_to_token(i)
        local = i - self.slot_start[slot]
        return f"⟦{slot}⟧" + self.slots[slot].decode([self._local2hf[slot][local]])

    def slot_of(self, i: int) -> str:
        return self._id_slot[i]
