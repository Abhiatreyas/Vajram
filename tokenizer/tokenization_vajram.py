"""
HuggingFace wrapper for the Vajram v3 partitioned tokenizer.

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

The encode/decode logic lives in vajram_tokenizer.py (shared with data prep); this class only adapts
it to the PreTrainedTokenizer interface. Token strings are OLMo's byte-level strings for ids < the
control range end, and "⟦SLOT⟧<byte-level piece>" for Indic slot ids, which keeps them unique.
"""

import os
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

try:
    from .vajram_tokenizer import SLOT_ORDER, VajramTokenizer
except ImportError:  # plain import when used outside HF dynamic modules
    from vajram_tokenizer import SLOT_ORDER, VajramTokenizer

VOCAB_FILES = {"layout_file": "vajram_layout.json", "olmo_file": "olmo_slot.json"}
VOCAB_FILES.update({f"slot_{s}": f"slots/{s}.json" for s in SLOT_ORDER})


class VajramTokenizerHF(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, layout_file=None, olmo_file=None, **kwargs):
        for s in SLOT_ORDER:
            kwargs.pop(f"slot_{s}", None)
        self.core = VajramTokenizer.from_dir(os.path.dirname(layout_file))
        self._vocab: Optional[Dict[str, int]] = None
        kwargs.setdefault("eos_token", "<|endoftext|>")
        kwargs.setdefault("pad_token", "<|pad|>")
        kwargs.setdefault("bos_token", None)
        super().__init__(**kwargs)

    # ---------------------------------------------------------------- vocab
    @property
    def vocab_size(self) -> int:
        return self.core.vocab_size

    def __len__(self) -> int:
        return self.core.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        if self._vocab is None:
            self._vocab = {self._convert_id_to_token(i): i for i in range(self.core.vocab_size)}
        return dict(self._vocab)

    def _convert_id_to_token(self, index: int) -> str:
        slot = self.core.slot_of(index)
        if slot == "OLMO":
            return self.core.olmo.id_to_token(index)
        local = index - self.core.slot_start[slot]
        return f"⟦{slot}⟧" + self.core.slots[slot].id_to_token(self.core._local2hf[slot][local])

    def _convert_token_to_id(self, token: str) -> int:
        if self._vocab is None:
            self.get_vocab()
        return self._vocab.get(token, self.core.layout["pad_token_id"])

    # ------------------------------------------------------------ encoding
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return [self._convert_id_to_token(i) for i in self.core.encode(text)]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.core.decode([self._convert_token_to_id(t) for t in tokens])

    def _decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids = [int(i) for i in token_ids]
        if skip_special_tokens:
            specials = set(self.all_special_ids)
            ids = [i for i in ids if i not in specials]
        return self.core.decode(ids, skip_special_tokens=skip_special_tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        import shutil
        src_dir = os.path.dirname(self.core_layout_path) if hasattr(self, "core_layout_path") else None
        out = []
        for key, rel in VOCAB_FILES.items():
            dst = os.path.join(save_directory, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if key == "layout_file":
                import json
                json.dump(self.core.layout, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            elif key == "olmo_file":
                self.core.olmo.save(dst)
            else:
                self.core.slots[key[len("slot_"):]].save(dst)
            out.append(dst)
        for f in ("vajram_tokenizer.py", "tokenization_vajram.py"):
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
            if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(os.path.join(save_directory, f)):
                shutil.copy2(src, os.path.join(save_directory, f))
        return tuple(out)
