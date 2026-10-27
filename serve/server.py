#!/usr/bin/env python3
"""Vajram — OpenAI-compatible chat endpoint with partitioned Indic tokenizer support.

Endpoints:
  - GET  /v1/models            OpenAI models list
  - POST /v1/chat/completions  OpenAI chat completion (streaming & non-streaming)
  - GET  /health               Health and readiness check

Environment variables:
  VAJRAM_BACKEND       Backend engine: "llama" (default, runs via llama-server) or "torch"
  VAJRAM_MODEL_DIR     Path to model weights or HF snapshot directory
  VAJRAM_TOKENIZER_DIR Path to directory with layout/slots/tokenizer files (default: VAJRAM_MODEL_DIR or ../tokenizer)
  VAJRAM_LLAMA_URL     URL of llama-server (default: http://127.0.0.1:8090)
  VAJRAM_PORT          Port to bind server (default: 8080)
  VAJRAM_HOST          Host address to bind server (default: 0.0.0.0)
  VAJRAM_API_KEY       Bearer token for authentication (empty = open/no auth)
  VAJRAM_MODEL_NAME    Model name returned by /v1/models (default: vajram-8b)
  VAJRAM_MODEL_ROUTING Pure model tool routing: 1 (default) or 0
  VAJRAM_TOOLS         Enable tool calling: 1 (default) or 0
  VAJRAM_SLOTS         Concurrent slots for llama backend (default: 1)
  VAJRAM_MAX_CTX       Max context length (default: 16384)
  VAJRAM_CUTOFF        Knowledge cutoff date string (e.g. "December 2024")
  VAJRAM_KEEP_SCRIPT   Align reply script to user's language: 1 or 0 (default: 0)
  VAJRAM_UNSAFE_PYTHON Enable arbitrary Python subprocess execution: 1 or 0 (default: 0)
  SCRIPTURE_DIR        Directory containing scripture store (default: ../scripture)
"""
import json, urllib.request, os, re, sys, threading, time, uuid

# Configure stdout and stderr for UTF-8 so Indic scripts print reliably on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# transformers' threaded weight loader intermittently access-violates on ROCm/Windows:
# load synchronously
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
import torch

GPU_CAP_GB = float(os.environ.get("VAJRAM_GPU_MEM_GB", "18"))
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory))

try:
    from . import tools
except ImportError:
    import tools

from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse, FileResponse

BACKEND = os.environ.get("VAJRAM_BACKEND", "llama")
LLAMA_URL = os.environ.get("VAJRAM_LLAMA_URL", "http://127.0.0.1:8090")
MODEL_DIR = os.environ.get("VAJRAM_MODEL_DIR", "")
_CANDIDATES = [
    os.environ.get("VAJRAM_TOKENIZER_DIR"),
    MODEL_DIR,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tokenizer")),
    r"E:\LLM_exp\release\hf",
    r"E:\models\vajram7b_r6c_dpo",
]
TOKENIZER_DIR = next((d for d in _CANDIDATES if d and os.path.exists(os.path.join(d, "vajram_layout.json"))),
                     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tokenizer")))
PORT = int(os.environ.get("VAJRAM_PORT", "8080"))
API_KEY = os.environ.get("VAJRAM_API_KEY", "")
MODEL_NAME = os.environ.get("VAJRAM_MODEL_NAME", "vajram-8b")
MODEL_ROUTING = os.environ.get("VAJRAM_MODEL_ROUTING", "1") == "1"
MAX_CTX = int(os.environ.get("VAJRAM_MAX_CTX", "16384"))
SLOTS = int(os.environ.get("VAJRAM_SLOTS", "1"))
DEFAULT_TEMPERATURE = float(os.environ.get("VAJRAM_TEMPERATURE", "0.3"))

# Add tokenizer path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tokenizer")))
import vajram_tokenizer as vt

print(f"loading tokenizer from {TOKENIZER_DIR} ...", flush=True)
tok = vt.VajramTokenizer.from_dir(TOKENIZER_DIR)
IM_END, EOS = tok.layout["im_end_id"], tok.layout["eos_token_id"]

def load_model():
    """Load model with Transformers (torch backend)."""
    if not MODEL_DIR or not os.path.isdir(MODEL_DIR):
        raise RuntimeError("VAJRAM_MODEL_DIR must point to a valid model directory when using BACKEND=torch")
    from transformers import AutoModelForCausalLM, AutoConfig
    if os.environ.get("VAJRAM_HF_LOAD") == "1":
        return AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
    
    import glob as _g, struct
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa")
    params = dict(m.named_parameters()) | dict(m.named_buffers())
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    seen = set()
    with torch.no_grad():
        for f in sorted(_g.glob(os.path.join(MODEL_DIR, "*.safetensors"))):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
                header.pop("__metadata__", None)
                base = 8 + n
                for k, v in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                    a, b = v["data_offsets"]
                    fh.seek(base + a)
                    buf = bytearray(fh.read(b - a))
                    if k in params:
                        t = torch.frombuffer(buf, dtype=dtypes[v["dtype"]]).view(v["shape"])
                        params[k].copy_(t)
                        seen.add(k)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
    if getattr(cfg, "tie_word_embeddings", False):
        m.tie_weights()
    missing = [k for k in dict(m.named_parameters()) if k not in seen]
    if missing:
        raise RuntimeError(f"weights missing from {MODEL_DIR}: {missing[:5]} ({len(missing)})")
    return m

if BACKEND == "llama":
    if not MODEL_ROUTING:
        raise SystemExit("VAJRAM_BACKEND=llama supports model routing only (VAJRAM_MODEL_ROUTING=1)")
    model = None
    print(f"backend: llama-server at {LLAMA_URL}", flush=True)
else:
    t0 = time.time()
    print(f"loading model weights from {MODEL_DIR} ...", flush=True)
    model = load_model().eval()
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"loaded in {time.time() - t0:.0f}s on {dev_name}", flush=True)

LOCK = threading.BoundedSemaphore(SLOTS) if BACKEND == "llama" else threading.Lock()
ROUTE_IDS = {tok.encode(t)[0] for t in tok.layout["control_tokens"] if t.startswith(("<|lang_", "<|act_"))}
LANG_IDS = {tok.encode(t)[0] for t in tok.layout["control_tokens"] if t.startswith("<|lang_")}
CLARIFY_ID = tok.encode("<|act_clarify|>")[0]
# r1c over-learned <|act_clarify|>: it fires on greetings ("Hi" -> "which city?"). Penalise it at the routing
# position (the token right after the language tag) so it only wins when the model is clearly unsure.
ACT_IDS = {a: tok.encode(f"<|act_{a}|>")[0] for a in ("direct", "think", "tool", "refuse", "clarify")}
# r1c initialised the typed-decision labels (<|opt_NN|>, <|yes|> ...) from digit / word embeddings, so in
# number-heavy text the model sometimes emits <|opt_05|> where a digit belongs ("6 6 6", "}5}5}5" garbage after
# search results, 2026-10-03). Normal generation may only use the control tokens that belong in a reply.
CONTROL_START, CONTROL_END = tok.layout["control_start"], tok.layout["control_end"]
IN_REPLY = {tok.encode(t)[0] for t in ("<think>", "</think>", "<tool_call>", "</tool_call>")}
BANNED_IDS = [i for i in range(CONTROL_START, CONTROL_END) if i not in IN_REPLY and i not in ROUTE_IDS]
BANNED = torch.tensor(BANNED_IDS, device="cuda") if model is not None else None
ROUTE_T = torch.tensor(sorted(ROUTE_IDS), device="cuda") if model is not None else None
TOOLS_ENABLED = os.environ.get("VAJRAM_TOOLS", "1") == "1"
MAX_TOOL_ROUNDS = 3
TOOL_ROUTE_MIN_P = float(os.environ.get("VAJRAM_TOOL_MIN_P", "0.5"))
SCRIPTURE_MIN_P = 0.35  # argmax of 4 options; a wrong call only returns "not found", a missed one recites a wrong verse
MEANING = {"kn": "**ಅರ್ಥ:** ", "en": "**Meaning:** "}
# pushes the action token away from <|act_clarify|>; 0 in pure model routing (the model alone decides)
CLARIFY_BIAS = float(os.environ.get("VAJRAM_CLARIFY_BIAS", "0" if os.environ.get("VAJRAM_MODEL_ROUTING", "1") == "1" else "-8"))
PREFILL_CHUNK = 512
FORCE_LANG = os.environ.get("VAJRAM_FORCE_LANG", "0") == "1"
PREFIX = re.compile(r"^(?:<\|(?:lang|act)_[a-z_]+\|>)+")
app = FastAPI()


def render(messages):
    s = ""
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):  # OpenAI content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        s += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return s + "<|im_start|>assistant\n"


class Visible:
    """Turns raw model text into what the user should see, incrementally."""

    def __init__(self):
        self.raw, self.sent, self.tags, self.tools = "", 0, None, []

    def feed(self, raw):
        self.raw = raw
        if self.tags is None:  # wait until the routing prefix is complete
            m = PREFIX.match(raw)
            if not m or (raw[m.end():m.end() + 2] == "<|" and len(raw) < m.end() + 30):
                if not raw.startswith("<|") or len(raw) > 60:
                    self.tags = ""
                else:
                    return ""
            else:
                self.tags = m.group(0)
        body = raw[len(self.tags):]
        body = re.sub(r"<think>.*?</think>", "", body, flags=re.S)
        if "<think>" in body:  # still thinking
            body = body[: body.index("<think>")]
        for call in re.findall(r"<tool_call>(.*?)</tool_call>", body, flags=re.S):
            if call not in self.tools:
                self.tools.append(call)
        body = re.sub(r"<tool_call>.*?</tool_call>", "", body, flags=re.S)
        if "<tool_call>" in body:
            body = body[: body.index("<tool_call>")]
        cut = body.rfind("<")  # hold back a possibly unfinished tag
        safe = body if cut == -1 or ">" in body[cut:] else body[:cut]
        out, self.sent = safe[self.sent:], max(self.sent, len(safe))
        return out


# r4 maths habit (suite 2026-10-22): the worked solution sits in <think> and the visible reply is only "\boxed{x}"
# (70% of r4's translated maths rows). Then the solution is shown before the answer; the first HOLD visible characters
# after </think> wait so this also works when streaming.
BOXED_ONLY = re.compile(r"^\$*\\boxed\{(?:[^{}]|\{[^{}]*\})*\}\$*\.?$")  # one brace level: \boxed{3.5 \text{ hours}}
HOLD = 80


def with_working(raw, visible):
    """a visible reply that is only a boxed answer -> the thought's worked solution, then the answer"""
    if "<think>" not in raw or not BOXED_ONLY.match(visible.strip()):
        return visible
    thought = raw.split("<think>", 1)[1].split("</think>")[0].strip()
    return f"{thought}\n\n{visible.strip()}" if thought else visible


SCRIPTS = {"kn": (0x0C80, 0x0CFF), "te": (0x0C00, 0x0C7F), "ta": (0x0B80, 0x0BFF), "ml": (0x0D00, 0x0D7F),
           "bn": (0x0980, 0x09FF)}  # Devanagari is hi or mr, so it is left to the model
LANG_TAG = {l: tok.encode(f"<|lang_{l}|>")[0] for l in SCRIPTS}


# Identity (GCP deployment, 2026-10-22): untold, r4 says it is GPT-4 / Claude / ChatGPT (from its training data).
# VAJRAM_IDENTITY is prepended to the system message; English only, because Kannada text in the system prompt pulls
# English answers into Kannada. The native spelling of the name is added only for a Kannada / Devanagari user, else
# the model invents one ("ವಲ್ರಾಮ್"). Empty (default) = no identity line, as in the eval runs.
IDENTITY = os.environ.get("VAJRAM_IDENTITY", "")
NAME_SPELLING = {"kn": "In Kannada your name is written ವಜ್ರಮ್ and the company ಲ್ಯಾಂಗ್‌ಬ್ರೇನ್ಸ್ AI.",
                 "deva": "In Hindi your name is written वज्रम."}


# Reply language follows the user's latest message (GCP deployment, 2026-10-22: after Kannada turns, an English
# "check the web for latest mac releases" was answered in Kannada). Penalising tags did not work: pushed off Kannada the
# model also changed its action and said "I cannot browse the web". So the model routes freely; only a direct answer
# whose language tag breaks the rule is restarted with the right tag (nothing has been shown yet), and the answer after
# a tool call starts with the right tag (the model still picks the action). Latin script is Kanglish / Hinglish / ...
# only with romanized-Indic words, else English. A message that names a language or asks for a translation is left to
# the model. VAJRAM_KEEP_SCRIPT=1; off by default (eval runs).
KEEP_SCRIPT = os.environ.get("VAJRAM_KEEP_SCRIPT", "0") == "1"
NAMES_LANGUAGE = re.compile(r"(?i)english|inglish|kannad|kanglish|hindi|telugu|tamil|malayalam|marathi|bengali|bangla|"
                            r"translat|ಕನ್ನಡ|ಇಂಗ್ಲಿಷ್|ಆಂಗ್ಲ|ಹಿಂದಿ|ತೆಲುಗು|ತಮಿಳು|ಮಲಯಾಳ|ಮರಾಠಿ|ಬಂಗಾಳಿ|ಅನುವಾದ|"
                            r"हिंदी|हिन्दी|अंग्रेज़ी|अंग्रेजी|कन्नड़|मराठी|अनुवाद")
KANNADA_SCRIPT_ASKED = re.compile(r"(?i)kannada\s+(script|lipi|letters)|ಕನ್ನಡ\s*ಲಿಪಿ|ಕನ್ನಡ\s*ಅಕ್ಷರ")
ROMAN_MARKERS = {  # common romanized words per language: their presence makes Latin text Kanglish / Hinglish / ...
    "kn_rom": "illa beku beda maadi madi enu yenu enri hege hegide yaaru yavaga yaavaga ninna nimma nanage naanu alli "
              "idu adu heli helri gottilla gottu swalpa sari howdu houdu banni irutte ide aagide kodi idanna eshtu ondu",
    "hi_rom": "hai hain kya nahi nahin kaise kaisa mujhe aap tum kyun kab kahan karo kijiye batao bataiye mera meri",
    "te_rom": "emi enti ela ledu cheyyi cheppu meeru nenu kavali undi",
    "ta_rom": "enna eppadi illai sollu naan neenga vendum irukku"}
ROMAN_MARKERS = {k: set(v.split()) for k, v in ROMAN_MARKERS.items()}
NATIVE = {"kn": ("kn", {"kn"}), "deva": ("hi", {"hi", "mr"}), "te": ("te", {"te"}), "ta": ("ta", {"ta"}),
          "ml": ("ml", {"ml"}), "bn": ("bn", {"bn"})}
SCRIPT_RANGES = {**SCRIPTS, "deva": (0x0900, 0x097F)}
LANG_ID = {l: tok.encode(f"<|lang_{l}|>")[0] for l in
           ("en", "kn", "kn_rom", "hi", "hi_rom", "te", "te_rom", "ta", "ta_rom", "ml", "ml_rom", "mr", "mr_rom", "bn", "bn_rom")}


ID_LANG = {i: l for l, i in LANG_ID.items()}
REPLY_IN = {"en": "English", "kn": "Kannada (Kannada script)", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
            "ml": "Malayalam", "bn": "Bengali",
            "kn_rom": "Kanglish (Kannada in English letters, like the user)",
            "hi_rom": "Hinglish (Hindi in English letters, like the user)",
            "te_rom": "Telugu in English letters, like the user", "ta_rom": "Tamil in English letters, like the user"}


class LangRestart(Exception):
    pass


def with_reply_language(messages, tag_id):
    """the messages with "Reply in <language>." added to the system message (the tag alone loses to the history)"""
    line = f"Reply in {REPLY_IN.get(ID_LANG.get(tag_id), 'the language of the latest user message')}."
    if messages and messages[0].get("role") == "system":
        return [{**messages[0], "content": (messages[0].get("content") or "") + "\n" + line}] + messages[1:]
    return [{"role": "system", "content": line}] + messages


def reply_lang_rule(messages):
    """(allowed language-tag ids, default tag id) for the reply, from the latest user message; None = model decides"""
    if not KEEP_SCRIPT:
        return None
    text = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    if not isinstance(text, str):
        return None
    if KANNADA_SCRIPT_ASKED.search(text):  # "in Kannada script": the model tends to answer in Kanglish
        return {LANG_ID["kn"]}, LANG_ID["kn"]
    if NAMES_LANGUAGE.search(text):
        return None
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    share = {s: sum(lo <= ord(c) <= hi for c in letters) / len(letters) for s, (lo, hi) in SCRIPT_RANGES.items()}
    native = max(share, key=share.get)
    # native script counts from a quarter of the letters: Kannada questions are full of English terms ("GPU", "FP8")
    if share[native] >= 0.25:
        default, allowed = NATIVE[native]
        return {LANG_ID[l] for l in allowed}, LANG_ID[default]
    if sum(c.isascii() for c in letters) / len(letters) <= 0.5:
        return None
    words = set(re.findall(r"[a-z]+", text.lower()))
    hits = {l: len(words & m) for l, m in ROMAN_MARKERS.items()}
    best = max(hits, key=hits.get)
    if hits[best] == 0:
        return {LANG_ID["en"]}, LANG_ID["en"]
    return {i for l, i in LANG_ID.items() if l == "en" or l.endswith("_rom")}, LANG_ID[best]


# r5 (2026-10-22) is trained with "Knowledge cutoff: <month year>." right after the date line, so it can tell what may
# have changed since. VAJRAM_CUTOFF="December 2024" for r5; unset = no line (r4 and earlier, eval runs).
CUTOFF_LINE = f"\nKnowledge cutoff: {os.environ['VAJRAM_CUTOFF']}." if os.environ.get("VAJRAM_CUTOFF") else ""


def identity_line(messages):
    if not IDENTITY:
        return ""
    lang = script_lang(messages)
    if lang is None:
        text = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
        letters = [c for c in text if c.isalpha()] if isinstance(text, str) else []
        if letters and sum(0x0900 <= ord(c) <= 0x097F for c in letters) / len(letters) > 0.5:
            lang = "deva"
    return IDENTITY.strip() + (" " + NAME_SPELLING[lang] if lang in NAME_SPELLING else "") + "\n"


def script_lang(messages):
    """Language of the last user message by script (> 50% of its letters), else None (model decides)."""
    text = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    letters = [c for c in text if c.isalpha()] if isinstance(text, str) else []
    for l, (lo, hi) in SCRIPTS.items():
        if letters and sum(lo <= ord(c) <= hi for c in letters) / len(letters) > 0.5:
            return l
    return None


def _llama(body, stream=False):
    req = urllib.request.Request(LLAMA_URL + "/completion", json.dumps({**body, "stream": stream}).encode(),
                                 {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    if not stream:
        return json.load(r)

    def events():
        try:
            for x in r:
                line = x.decode("utf-8").strip()
                if line.startswith("data:"):
                    yield json.loads(line[5:])
        finally:  # closing the connection makes llama-server stop generating for this request
            r.close()
    return events()


def ev_tokens(ev):
    """token ids of one llama-server stream event. llama-server holds a token back while it ends inside a UTF-8 character
    and then sends only the LAST token id with the completed text (2026-10-22: "10⁸" = 53233 + 116 arrived as 116 alone ->
    "10<U+FFFD>" in the chat). The completed text is right for these byte tokens (Indic slot tokens are never held back),
    so the event's ids are re-encoded from it."""
    toks = ev.get("tokens") or []
    c = ev.get("content") or ""
    if len(toks) == 1 and c and "�" not in c and "<|vj_" not in c and "�" in tok.decode(toks):
        return tok.encode(c)
    return toks


def llama_generate(ids, max_tokens, temperature, top_p, clarify_bias, force_act, force_lang, prefill, lang_rule=None,
                   lead=()):
    """Same decoding rules as the torch loop: banned control ids everywhere, language tag forced from the user's script
    at position 0, clarify bias / forced action at position 1, no routing tags after that. llama.cpp's logit_bias
    applies to every position, so positions 0 and 1 are single-token requests (the prompt stays cached)."""
    stop = (IM_END, EOS)
    base = {"cache_prompt": True, "return_tokens": True, "temperature": max(temperature, 0.0), "top_p": top_p,
            "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.1 if temperature > 0 else 1.0, "repeat_last_n": 128}
    ban = [[i, False] for i in BANNED_IDS]
    out_ids = []
    if lead and not prefill:
        # reply language fixed by reply_lang_rule: the tag is given, the model picks the action (another tool or the
        # answer) in one single-token step, then the reply streams with routing tags banned
        out_ids = list(lead)
        r = _llama({**base, "prompt": ids + out_ids, "n_predict": 1,
                    "logit_bias": ban + [[i, False] for i in sorted(ROUTE_IDS - set(ACT_IDS.values()))]})
        t = (r.get("tokens") or [None])[0]
        if t is None or t in stop:
            return
        out_ids.append(t)
        yield out_ids, len(ids)
        prefill = "lead"  # continue below as after a prefill: no routing tags from here on
    elif not prefill and force_lang is None and force_act not in ACT_IDS and clarify_bias == 0:
        # nothing to force at positions 0 / 1 (model routing defaults): ONE streamed request instead of three. Each
        # llama-server request restores the sliding-window cache checkpoint (~0.3 s on the L4, 2026-10-22), so this
        # saves ~0.6 s per reply and keeps concurrent users from stalling each other. Routing tags are allowed where
        # the 3-request path allows them (position 0; position 1 after a language tag) and dropped anywhere else.
        stream, restart = _llama({**base, "prompt": ids, "n_predict": max_tokens, "logit_bias": ban}, stream=True), False
        try:
            for ev in stream:
                for t in ev_tokens(ev):
                    if t in stop:
                        return
                    if t in ROUTE_IDS and not (not out_ids or (len(out_ids) == 1 and out_ids[0] in LANG_IDS)):
                        continue
                    out_ids.append(t)
                    if (lang_rule and len(out_ids) == 2 and out_ids[0] in LANG_IDS and out_ids[0] not in lang_rule[0]
                            and out_ids[1] != ACT_IDS["tool"]):
                        restart = True  # a direct answer in the wrong language: nothing shown yet, redo it
                        break
                    yield out_ids, len(ids)
                if restart or ev.get("stop"):
                    break
        finally:
            stream.close()
        if restart:
            print(f"reply language: {tok.decode(out_ids[:1])} -> {tok.decode([lang_rule[1]])}", flush=True)
            raise LangRestart  # generate() redoes it with the tag and a language instruction
        return
    if not prefill:
        for pos in (0, 1):
            bias = list(ban)
            if pos == 0 and force_lang is not None:
                bias.append([force_lang, 30.0])
            if pos == 1:
                if out_ids[0] not in LANG_IDS:
                    break
                bias.append([CLARIFY_ID, clarify_bias])
                if force_act in ACT_IDS:
                    bias.append([ACT_IDS[force_act], 30.0])
            r = _llama({**base, "prompt": ids + out_ids, "n_predict": 1, "logit_bias": bias})
            t = (r.get("tokens") or [None])[0]
            if t is None or t in stop:
                return
            out_ids.append(t)
            yield out_ids, len(ids)
    rest = max_tokens - len(out_ids)
    if rest <= 0:
        return
    for ev in _llama({**base, "prompt": ids + out_ids, "n_predict": rest,
                      "logit_bias": ban + [[i, False] for i in sorted(ROUTE_IDS)]}, stream=True):
        for t in ev_tokens(ev):
            if t in stop:
                return
            out_ids.append(t)
            yield out_ids, len(ids)
        if ev.get("stop"):
            return


TC_OPEN, TC_CLOSE = tok.encode("<tool_call>")[0], tok.encode("</tool_call>")[0]


def fix_tool_open(ids, depth=0):
    """r6 (CPT-base SFT) has near-identical <tool_call> / </tool_call> output rows (cosine 0.997): when sampling it
    sometimes opens a call with </tool_call> (2026-10-25: Gita 6.7 lookup shown as raw text). A closing tag while no
    call is open can only be an opening tag: read it as one."""
    out = None
    for i, t in enumerate(ids):
        if t == TC_CLOSE and depth == 0:
            out = out or list(ids)
            out[i] = TC_OPEN
            depth = 1
        elif t == TC_OPEN:
            depth = 1
        elif t == TC_CLOSE:
            depth = 0
    return out or ids


def generate(messages, max_tokens, temperature, top_p, clarify_bias=CLARIFY_BIAS, force_act=None, prefill="", lang_rule=None,
             lead=()):
    depth0 = 1 if prefill.rfind("<tool_call>") > prefill.rfind("</tool_call>") else 0  # server-prefilled call already open
    logged = False
    for out_ids, n in _generate(messages, max_tokens, temperature, top_p, clarify_bias, force_act, prefill, lang_rule, lead):
        fixed = fix_tool_open(out_ids, depth0)
        if fixed is not out_ids and not logged:  # counted by the row-fix A/B (rowfix/run_rowfix.sh)
            print("fixed tool open", flush=True)
            logged = True
        yield fixed, n


def _generate(messages, max_tokens, temperature, top_p, clarify_bias=CLARIFY_BIAS, force_act=None, prefill="", lang_rule=None,
              lead=()):
    ids = tok.encode(render(messages) + prefill)[-(MAX_CTX - max_tokens):]
    # owner 2026-10-04: the model picks the reply language (system-prompt instruction, else the user's language).
    # VAJRAM_FORCE_LANG=1 restores forcing the tag from the user's script, only to reproduce earlier eval runs.
    force_lang = LANG_TAG.get(script_lang(messages)) if FORCE_LANG and not prefill else None
    if BACKEND == "llama":
        if lead:
            ids = tok.encode(render(with_reply_language(messages, lead[0])) + prefill)[-(MAX_CTX - max_tokens):]
        try:
            yield from llama_generate(ids, max_tokens, temperature, top_p, clarify_bias, force_act, force_lang, prefill,
                                      lang_rule, lead)
        except LangRestart:
            yield from _generate(messages, max_tokens, temperature, top_p, clarify_bias, force_act, prefill, None,
                                 (lang_rule[1],))
        return
    past, out_ids, n_prompt = None, [], len(ids)
    with torch.inference_mode():
        # chunked prefill, logits for the last position only: a 3k-token prompt in one pass needs ~0.5 GB for the
        # full-sequence logits alone (OOM at the 17 GiB cap, 2026-10-04 pure-routing eval)
        for i in range(0, len(ids) - 1, PREFILL_CHUNK):
            past = model(input_ids=torch.tensor([ids[i:min(i + PREFILL_CHUNK, len(ids) - 1)]], device="cuda"),
                         past_key_values=past, use_cache=True, logits_to_keep=1).past_key_values
        x = torch.tensor([ids[-1:]], device="cuda")
        for _ in range(max_tokens):
            o = model(input_ids=x, past_key_values=past, use_cache=True, logits_to_keep=1)
            past, logits = o.past_key_values, o.logits[0, -1].float()
            logits[BANNED] = -float("inf")  # decision labels / reserved / image tokens never appear in a reply
            if prefill or len(out_ids) >= 2:  # routing tags only at the first two positions
                logits[ROUTE_T] = -float("inf")
            if not out_ids and force_lang is not None:  # language tag = script of the user message
                logits[force_lang] += 30.0
            if not prefill and len(out_ids) == 1 and out_ids[0] in LANG_IDS:  # routing (action) position
                logits[CLARIFY_ID] += clarify_bias
                if force_act in ACT_IDS:
                    logits[ACT_IDS[force_act]] += 30.0
            if out_ids and temperature > 0:  # repetition penalty 1.1 when sampling (greedy + penalty garbles copies)
                prev = torch.tensor(list(set(out_ids)), device="cuda")
                lp = logits[prev]
                logits[prev] = torch.where(lp > 0, lp / 1.1, lp * 1.1)
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                probs = torch.softmax(logits / temperature, -1)
                sp, si = probs.sort(descending=True)
                keep = sp.cumsum(0) - sp < top_p
                nxt = int(si[keep][torch.multinomial(sp[keep] / sp[keep].sum(), 1)])
            if nxt in (IM_END, EOS):
                break
            out_ids.append(nxt)
            yield out_ids, n_prompt
            x = torch.tensor([[nxt]], device="cuda")


CTX_WORDS = {"kn": ("ಸಂದರ್ಭ", "ಪ್ರಶ್ನೆ"), "en": ("Context", "Question"), "te": ("సందర్భం", "ప్రశ్న"),
             "ta": ("சூழல்", "கேள்வி"), "ml": ("സന്ദർഭം", "ചോദ്യം"), "bn": ("প্রসঙ্গ", "প্রশ্ন")}
DEC_SYS = ("Decision mode: answer with exactly one label token. If the state does not contain enough information "
           "to decide, answer <|unknown|>.")
FRESH = re.compile(r"(?i)\b(today|now|current|latest|recent|news|weather|price|rate|score|live|this (week|month|year)|2025|2026)\b"
                   r"|ಇಂದು|ಈಗ|ಇತ್ತೀಚ|ಸುದ್ದಿ|ಹವಾಮಾನ|ಬೆಲೆ|ದರ|ಸ್ಕೋರ್|ಈ ವರ್ಷ|आज|मौसम|ख़बर|खबर|कीमत")
ROUTE_OPTIONS = ["arithmetic or a calculation with numbers (use a calculator)",
                 "looking up facts: current or live information (weather, news, prices, scores, recent events) or "
                 "exact specifications, figures and details of products, people, places or organisations (search the web)",
                 "the exact text of a verse, shloka or mantra from a scripture such as the Bhagavad Gita or an Upanishad "
                 "(look it up in the verified collection)",
                 "neither: conversation, explanation, writing, translation or general knowledge"]
OPT_IDS = [tok.encode(f"<|opt_{i:02d}|>")[0] for i in range(len(ROUTE_OPTIONS))]


def decide_tool(messages):
    """Vajram's own typed-decision mode (92-95% on held-out intents) picks the tool route: one forward pass."""
    user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    if not isinstance(user, str) or not user.strip():
        return None
    q = "\n".join(f"<|opt_{i:02d}|> {o}" for i, o in enumerate(ROUTE_OPTIONS))
    prompt = (f"<|im_start|>system\n{DEC_SYS}<|im_end|>\n<|im_start|>user\n<state>{user[:2000]}</state>\n"
              f"Question: What does answering this request need?\n{q}<|im_end|>\n<|im_start|>assistant\n<|decide|>")
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([tok.encode(prompt)], device="cuda")).logits[0, -1].float()
    probs = torch.softmax(logits[OPT_IDS], -1).tolist()
    pick = max(range(len(probs)), key=probs.__getitem__)
    print(f"decide {[round(x, 2) for x in probs]} -> {pick} | {user[:50]!r}", flush=True)
    # Conservative routing (2026-10-04 eval: after the RoPE fix the first decision sends most prompts to "look up",
    # and injected search text then replaced the user's own passage / question). Every route needs its own yes/no.
    if re.search(r"https?://\S+", user) and pick != 2 and reads_page(user):
        return "fetch_page"
    if pick == 0 and probs[0] >= 0.28 and re.search(r"\d", user) and needs_calculation(user):
        return decide_sub(user, CALC_OPTIONS, "What does this calculation need?")
    if pick == 1 and needs_lookup(user) and not has_own_context(user):
        return decide_sub(user, LIVE_OPTIONS, "Which kind of information does this request need?")
    if pick == 2 and probs[2] >= SCRIPTURE_MIN_P and asks_for_verse(user):
        return "scripture_lookup"
    return None


def reads_page(user):
    return yes_no(user, "Does the user ask to read, open or summarise the web page at the link they gave?") >= 0.5


def needs_calculation(user):
    return yes_no(user, "Does answering need an exact arithmetic calculation (or a currency conversion) with the "
                        "numbers in the message?") >= 0.5


def has_own_context(user):
    """The user pasted the material to work on (a passage, code, a text to translate or rewrite): answer from it."""
    return bool(re.search(r"(?im)^\s*(context|passage|text|ಸಂದರ್ಭ|ಪಠ್ಯ)\s*:|```", user)) or len(user) > 600


YES_NO = [tok.encode(t)[0] for t in ("<|yes|>", "<|no|>")]


def yes_no(user, question):
    """Typed yes/no decision (<|yes|> / <|no|> label tokens): P(yes)."""
    prompt = (f"<|im_start|>system\n{DEC_SYS}<|im_end|>\n<|im_start|>user\n<state>{user[:2000]}</state>\n"
              f"Question: {question}<|im_end|>\n<|im_start|>assistant\n<|decide|>")
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([tok.encode(prompt)], device="cuda")).logits[0, -1].float()
    p = torch.softmax(logits[YES_NO], -1)[0].item()
    print(f"yes_no {p:.2f} | {question[:50]}", flush=True)
    return p


def needs_lookup(user):
    return yes_no(user, "Does answering this message need information that should be looked up (today's date or "
                        "time, weather, exchange rates, news, prices, exact specifications or figures of a product, or "
                        "facts about a specific person, place or organisation)? A greeting, small talk, creative "
                        "writing, translation or an explanation of a general concept does not.") >= 0.5


def asks_for_verse(user):
    return yes_no(user, "Is the user asking for the text of a specific verse, shloka or mantra from a scripture?") >= 0.5


LIVE_OPTIONS = [("weather", "the weather or temperature in a place"),
                ("currency_convert", "converting money between currencies, or an exchange rate"),
                ("get_datetime", "today's date, the day of the week, or the current time"),
                ("web_search", "news, prices, scores, recent events, facts about a specific person, place or "
                               "organisation, or any other information to look up"),
                (None, "no information to look up: a greeting, small talk, writing, translation or an opinion")]
CALC_OPTIONS = [("calculator", "arithmetic on numbers given in the message"),
                ("currency_convert", "converting money between currencies (needs today's exchange rate)")]


def decide_sub(user, options, question):
    """Second typed decision: which specific tool (or None). One forward pass, same format as decide_tool."""
    ids = [tok.encode(f"<|opt_{i:02d}|>")[0] for i in range(len(options))]
    q = "\n".join(f"<|opt_{i:02d}|> {d}" for i, (_, d) in enumerate(options))
    prompt = (f"<|im_start|>system\n{DEC_SYS}<|im_end|>\n<|im_start|>user\n<state>{user[:2000]}</state>\n"
              f"Question: {question}\n{q}<|im_end|>\n<|im_start|>assistant\n<|decide|>")
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([tok.encode(prompt)], device="cuda")).logits[0, -1].float()
    probs = torch.softmax(logits[ids], -1).tolist()
    pick = max(range(len(probs)), key=probs.__getitem__)
    print(f"decide_sub {[round(x, 2) for x in probs]} -> {options[pick][0]}", flush=True)
    return options[pick][0]


# first argument the model writes after the prefilled tool name (it continues the JSON; the call is closed at '"}')
FIRST_ARG = {"calculator": '"expression": "', "scripture_lookup": '"text": "', "weather": '"city": "',
             "currency_convert": '"amount": ', "get_datetime": '"timezone": "', "wikipedia": '"query": "',
             "fetch_page": '"url": "', "web_search": '"query": "'}
GROUNDED = {"web_search", "weather", "wikipedia", "fetch_page"}  # answered in the grounded-QA (context) format


LOOP_GUARD = os.environ.get("VAJRAM_LOOP_GUARD", "v2")  # v1 = the guard before 2026-10-14
LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)]|#+)\s*")


def repeated_line(raw):
    """Loop guard: the last finished line (>= 12 characters) was already written once.
    v2 (2026-10-14 suite review): lines inside ``` code blocks are not compared (code repeats lines: a second version
    of a function was cut at 'def is_palindrome(s'), and list markers are stripped first ('4. X' and '5. X' are the
    same line: a numbered Kanglish recipe looped to step 9 before any guard fired)."""
    lines = raw.split("\n")[:-1]
    if LOOP_GUARD == "v1":
        done = [l.strip() for l in lines if len(l.strip()) >= 12]
        return bool(done) and done.count(done[-1]) >= 2
    done, fence = [], False
    for l in lines:
        if l.strip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        t = LIST_MARK.sub("", l).strip()
        if len(t) >= 12:
            done.append(t)
    return bool(done) and done.count(done[-1]) >= 2


def token_loop(ids, max_n=16, reps=4):
    """True when the last tokens are one n-gram (n <= max_n) repeated `reps` times in a row."""
    for n in range(1, max_n + 1):
        if len(ids) >= n * reps and all(ids[-n * (r + 1):len(ids) - n * r] == ids[-n:] for r in range(reps)):
            if set(tok.decode(ids[-n:])) <= TABLE_CHARS:  # "|---|---|---|" table rules, "----" lines, "***"
                continue
            return True
    return False


TABLE_CHARS = set("|-:= \n*#_~.")


def numbers_from_user(args, convo):
    """Every number in a calculator expression must appear in the user's last message (catches mis-copies like
    12500.75 for 12450.75). Small constants (factors such as 0.18, 1.05, 100) are allowed."""
    expr = str((args or {}).get("expression", "")) if isinstance(args, dict) else str(args)
    user = next((m.get("content") or "" for m in reversed(convo) if m.get("role") == "user"), "")
    have = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(user).replace(",", ""))}
    for x in re.findall(r"\d+(?:\.\d+)?", expr.replace(",", "")):
        v = float(x)
        if v not in have and v > 100 and not any(abs(v - h * k) < 1e-6 for h in have for k in (0.01, 1)):
            return False
    return True


def repair_numbers(args, convo):
    """Snap each large number the model mis-copied to the nearest number in the user's message (within 5%)."""
    expr = str((args or {}).get("expression", ""))
    user = next((m.get("content") or "" for m in reversed(convo) if m.get("role") == "user"), "")
    have = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(user).replace(",", ""))]

    def fix(m):
        v = float(m.group(0))
        if v <= 100 or v in have or not have:
            return m.group(0)
        near = min(have, key=lambda h: abs(h - v))
        return (f"{near:.10g}" if abs(near - v) <= 0.05 * max(v, 1) else m.group(0))
    return {"expression": re.sub(r"\d+(?:\.\d+)?", fix, expr.replace(",", ""))}


def check_auth(authorization):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid API key")


UI_HTML_PATH = os.path.join(os.path.dirname(__file__), "ui.html")

def get_ui_html() -> str:
    if os.path.exists(UI_HTML_PATH):
        with open(UI_HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<html><body style='font-family:sans-serif;background:#0f1420;color:#f1f5f9;padding:40px;'><h1>Vajram API</h1><p>Vajram 8.4B Inference Server is running.</p></body></html>"


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def index():
    return HTMLResponse(get_ui_html())


@app.get("/logo-symbol.png")
@app.get("/static/logo-symbol.png")
def logo_symbol():
    path = os.path.join(os.path.dirname(__file__), "logo-symbol.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    raise HTTPException(404, "Logo not found")


@app.get("/logo.svg")
@app.get("/static/logo.svg")
def logo_svg():
    path = os.path.join(os.path.dirname(__file__), "logo.svg")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/svg+xml")
    raise HTTPException(404, "Logo not found")


@app.get("/v1", response_class=HTMLResponse)
@app.get("/v1/", response_class=HTMLResponse)
def v1_index():
    return RedirectResponse(url="/")


@app.get("/health")
def health():
    backend_status = "ready"
    if BACKEND == "llama":
        try:
            req = urllib.request.Request(LLAMA_URL + "/health")
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status != 200:
                    backend_status = f"llama-server status {r.status}"
        except Exception as e:
            backend_status = f"llama-server not ready ({e.__class__.__name__})"
    return {
        "status": "ok" if backend_status == "ready" else "degraded",
        "backend": BACKEND,
        "backend_status": backend_status,
        "model": MODEL_NAME
    }


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "vajram"}]}


SERVER_TOOL_NAMES = {t["function"]["name"] for t in tools.TOOLS + tools.R2_TOOLS + tools.SERVER_TOOLS}


def openai_calls(parsed, stream=False):
    """(name, args) pairs -> OpenAI tool_calls (stream chunks carry an index)."""
    out = []
    for i, (name, args) in enumerate(parsed):
        c = {"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
             "function": {"name": name, "arguments": json.dumps(args if isinstance(args, dict) else {"input": args}, ensure_ascii=False)}}
        out.append({"index": i, **c} if stream else c)
    return out


def client_tools(msgs, tool_defs, max_tokens, temperature, top_p, clarify_bias, cid, created, force_act=None):
    """One model turn with caller-defined tools, in Vajram's tool-use SFT format: the tools are declared in the system
    prompt, earlier calls are rendered as <|act_tool|><tool_call>{json}</tool_call>, results as `tool` turns. No
    server-side routing, prefill or repair, so this measures what the model itself does (agent tests)."""
    lang = script_lang(msgs) or "en"
    # same system layout as chat mode (and the round-3 training rows): tool block, date line, then the caller's system
    date_line = f"Current date and time: {tools.get_datetime('Asia/Kolkata')['datetime']} (India Standard Time)." + CUTOFF_LINE
    user_sys = "\n\n".join(str(m.get("content") or "") for m in msgs if m.get("role") == "system")
    sys_extra = date_line + ("\n" + user_sys if user_sys else "")
    convo = [{"role": "system", "content": "You can call tools. Available tools (JSON schema):\n"
              + json.dumps(tool_defs, ensure_ascii=False) + ("\n\n" + sys_extra if sys_extra else "")}]
    for m in msgs:
        if m.get("role") == "system":
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = "".join("<tool_call>" + json.dumps({"name": c["function"]["name"], "arguments": json.loads(
                c["function"].get("arguments") or "{}")}, ensure_ascii=False) + "</tool_call>" for c in m["tool_calls"])
            convo.append({"role": "assistant", "content": f"<|lang_{lang}|><|act_tool|>{m.get('content') or ''}{calls}"})
        else:
            convo.append({"role": m.get("role", "user"), "content": m.get("content") or ""})
    t, out_ids, n_prompt, stop = time.time(), [], 0, ""
    with LOCK:
        for out_ids, n_prompt in generate(convo, max_tokens, temperature, top_p, clarify_bias, force_act,
                                          lang_rule=reply_lang_rule(msgs)):
            if token_loop(out_ids):
                stop = "loop"
                break
    raw = tok.decode(out_ids)
    tags = (PREFIX.match(raw).group(0) if PREFIX.match(raw) else "")
    body = re.sub(r"<think>.*?</think>", "", raw[len(tags):], flags=re.S)
    calls, bad = [], []
    for c in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", body, flags=re.S):
        try:
            j = json.loads(c)
            args = j.get("arguments", {})
            calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                          "function": {"name": j["name"], "arguments": args if isinstance(args, str)
                                       else json.dumps(args, ensure_ascii=False)}})
        except Exception:
            bad.append(c[:300])
    text = re.sub(r"<tool_call>.*?(</tool_call>|$)", "", body, flags=re.S).strip()
    if not calls:
        text = with_working(raw, text)
    print(f"client-tools {tags!r} calls={len(calls)} bad={len(bad)} {round(time.time() - t)}s", flush=True)
    return {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
            "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                         "message": {"role": "assistant", "content": text or None,
                                     **({"tool_calls": calls} if calls else {})}}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(out_ids)},
            "vajram": {"route": tags, "malformed_calls": bad, "stopped": stop or
                       ("max_tokens" if len(out_ids) >= max_tokens else "")}}


@app.post("/v1/chat/completions")
async def chat(req: Request, authorization: str = Header(default="")):
    check_auth(authorization)
    body = await req.json()
    msgs = body.get("messages") or []
    max_tokens = min(int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024), 2048)
    temperature = float(body.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(body.get("top_p", 0.9))
    clarify_bias = float(body.get("clarify_bias", CLARIFY_BIAS))
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

    def chunk(delta, finish=None, **extra):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

    if body.get("tools"):  # OpenAI-style client tools: the model alone decides; calls are returned, not executed
        # blocking generation runs in a worker thread so other requests are served meanwhile
        return JSONResponse(await run_in_threadpool(client_tools, msgs, body["tools"], max_tokens, temperature, top_p,
                                                    clarify_bias, cid, created, body.get("force_act")))
    use_tools = body.get("tools_enabled", TOOLS_ENABLED)
    act = body.get("force_act")  # testing: "think" / "tool" / "direct" forces the routing action

    def run():
        t, n_prompt, n_out = time.time(), 0, 0
        # current date and time in every request (standard for chat assistants): date questions need no tool
        now = tools.get_datetime("Asia/Kolkata")["datetime"]
        lang_rule = reply_lang_rule(msgs)  # from the user's own message, before tool results are folded in
        date_line = identity_line(msgs) + f"Current date and time: {now} (India Standard Time)." + CUTOFF_LINE
        msgs_d = [m for m in msgs if m.get("role") == "system"]
        if msgs_d:
            msgs_d = [{**msgs_d[0], "content": date_line + "\n" + (msgs_d[0].get("content") or "")}] + msgs_d[1:]
        else:
            msgs_d = [{"role": "system", "content": date_line}]
        convo, extra_sys = msgs_d + [m for m in msgs if m.get("role") != "system"], ""
        route_lang = None
        with LOCK:
            route_act, prefill = act, ""
            chosen = decide_tool(convo) if use_tools and act is None and not MODEL_ROUTING else None
            # Declare tools exactly as in Vajram's tool-use SFT data (merged with any system prompt). Server routing:
            # only the chosen tool, or none: with all 8 schemas (~3.3k chars of JSON) the 7B copies the schema text
            # ("arithmetic", '"},}"}}') instead of answering (2026-10-03).
            if use_tools and (MODEL_ROUTING or chosen or act == "tool"):
                extra = "\n\n".join(str(m.get("content") or "") for m in convo if m.get("role") == "system")
                extra_sys = extra
                sp = tools.system_prompt(extra, tool_list=tools.SERVER_TOOLS) if MODEL_ROUTING else \
                    tools.system_prompt(extra, names=[chosen] if chosen else None)
                convo = [{"role": "system", "content": sp}] + [m for m in convo if m.get("role") != "system"]
            if chosen:  # typed decision picked a tool: prefill the call, the model writes only the argument
                lang = script_lang(convo) or "en"
                prefill = (f"<|lang_{lang}|><|act_tool|><tool_call>{{\"name\": \"{chosen}\", "
                           f"\"arguments\": {{{FIRST_ARG.get(chosen, chr(34) + 'query' + chr(34) + ': ' + chr(34))}")
            answer_pre = ""  # after a verse is shown: the answer starts at "Meaning:" so the verse is not retyped
            for rnd in range(MAX_TOOL_ROUNDS + 1):
                vis, out_ids, held = Visible(), [], ""
                think_t0, think_done = None, False
                pre = "" if rnd else prefill
                apre = answer_pre if rnd else ""
                lead = ()
                if rnd and not apre and lang_rule:
                    # after a tool call the model often starts the answer without a language tag and drifts to the
                    # conversation's language: start it with the tool step's tag if allowed, else the default
                    lead = (route_lang if route_lang in lang_rule[0] else lang_rule[1],)
                # the tool call is an exact copy: greedy. Answers (incl. from tool results) sample at <= 0.3: greedy
                # degenerates into loops on long contexts.
                temp = 0.0 if pre else (min(temperature, 0.3) if rnd else temperature)
                for out_ids, n_p in generate(convo, max_tokens, temp, top_p, clarify_bias,
                                             (None if rnd else route_act) if MODEL_ROUTING else ("direct" if rnd else route_act), pre or apre,
                                             None if rnd else lang_rule, lead):
                    n_prompt = n_prompt or n_p
                    k = 0  # the routing prefix is control tokens: drop them by id, never shown to the user
                    while k < len(out_ids) and out_ids[k] in ROUTE_IDS:
                        k += 1
                    vis.tags = ""
                    raw = tok.decode(out_ids[k:]) if k < len(out_ids) else ""
                    if pre:  # prefilled tool call: the visible text starts after the routing tags
                        raw = pre[pre.index("<tool_call>"):] + tok.decode(out_ids)
                        if "</tool_call>" not in raw and ("\"}" in raw or "\n" in raw):
                            raw = raw.split("\"}")[0] + "\"}}</tool_call>"  # argument finished: close the call
                    elif apre:
                        raw = apre[apre.index("|>", apre.index("<|act_")) + 2:] + tok.decode(out_ids)
                    if raw.endswith("\ufffd"):  # incomplete UTF-8 character: wait for the next token
                        continue
                    piece = vis.feed(raw)
                    # thinking indicator (2026-10-22): the reasoning text stays hidden; the client is told when it
                    # starts and how long it took, so the chat can show "Thinking..." / "Thought for 12 s"
                    if think_t0 is None and "<think>" in raw:
                        think_t0 = time.time()
                        yield "thinking", {"status": "thinking"}
                    if think_t0 is not None and not think_done and "</think>" in raw:
                        think_done = True
                        yield "thinking", {"status": "done", "ms": round((time.time() - think_t0) * 1000)}
                    stop = ""
                    if repeated_line(raw):  # loop guard: a full line repeated -> stop
                        stop = "repeated line"
                    elif token_loop(out_ids):  # loop guard: a short token pattern repeated many times -> stop
                        stop = "token pattern"
                    else:
                        flat = re.sub(r"\d+", "", raw)
                        if len(flat) > 200 and flat.count(flat[-50:]) >= 3:  # same passage 3x, list numbers ignored
                            stop = "repeated passage"
                    if stop:  # logged, so a cut answer can be told apart from a finished one (2026-10-14)
                        print(f"loop guard ({LOOP_GUARD}): stopped on {stop} after {len(out_ids)} tokens: "
                              f"{raw.strip()[-80:]!r}", file=sys.stderr, flush=True)
                        break
                    if piece:
                        if held is not None and "</think>" in vis.raw and not vis.tools and vis.sent <= HOLD:
                            held += piece  # a short reply after thinking may be a bare \boxed{}: wait (with_working)
                        else:
                            yield "delta", (held or "") + piece
                            held = None
                n_out += len(out_ids)
                if not rnd:
                    route_lang = out_ids[0] if out_ids and out_ids[0] in LANG_IDS else None
                if think_t0 is not None and not think_done:  # stopped while thinking (loop guard / max tokens)
                    think_done = True
                    yield "thinking", {"status": "done", "ms": round((time.time() - think_t0) * 1000)}
                if held:
                    yield "delta", with_working(vis.raw, held)
                if vis.sent == 0 and "<think>" in vis.raw and not vis.tools:
                    thought = vis.raw.split("<think>", 1)[1].split("</think>")[0].strip()
                    if thought:
                        yield "delta", thought
                if out_ids:
                    print(f"route r{rnd} {(apre or pre)[:28]!r}+{tok.decode(out_ids[:2])!r} | {vis.raw[:90]!r}", flush=True)
                if not (use_tools and vis.tools) or rnd == MAX_TOOL_ROUNDS:
                    break
                # tools the server does not own (declared by the client in its system prompt): return the calls,
                # as OpenAI tool_calls, instead of "unknown tool" errors (2026-10-25: system-prompt custom tools 5/18)
                client_calls = [tools.parse_call(c) for c in vis.tools if (tools.parse_call(c)[0] or "") not in SERVER_TOOL_NAMES]
                if client_calls:
                    yield "client_calls", client_calls
                    break
                results, searched = [], False  # run the requested tool calls, feed the results back
                for call in vis.tools:
                    name, args = tools.parse_call(call)
                    if not MODEL_ROUTING and name == "calculator" and not numbers_from_user(args, convo):
                        fixed = repair_numbers(args, convo)
                        print(f"copy check failed: {args} -> repaired {fixed}", flush=True)
                        args = fixed
                    q = " ".join(str(v) for v in args.values()).strip() if isinstance(args, dict) else str(args)
                    yield "tool", {"tool": name or "tool", "query": q, "status": "running"}
                    res = tools.run_tool(name, args)
                    print(f"tool {name} {q!r} -> {json.dumps(res, ensure_ascii=False)[:160]}", flush=True)
                    yield "tool", {"tool": name or "tool", "query": q, "status": "done"}
                    if name == "scripture_lookup" and res.get("found"):  # verse shown verbatim, never retyped by the model
                        lang = script_lang(convo) or "en"
                        yield "delta", tools.verse_block(res, lang)
                        if not MODEL_ROUTING:
                            answer_pre = f"<|lang_{lang}|><|act_direct|>" + MEANING.get(lang, MEANING["en"])
                    if MODEL_ROUTING:  # round-2 format: plain tool turn, as in training
                        results.append(tools.format_result(name, res, r2=True))
                        continue
                    if name == "web_search" and res.get("results"):
                        searched = True
                        results.append("\n".join(f"{r['title']}: {r['snippet']}" for r in res["results"]))
                    elif name in GROUNDED and not res.get("error"):  # weather / wikipedia / web page -> context
                        searched = True
                        results.append(tools.format_result(name, res))
                    elif name == "scripture_lookup" and res.get("found"):
                        searched = True
                        results.append(tools.format_result(name, res))
                    else:
                        results.append(tools.format_result(name, res))
                said = (pre + tok.decode(out_ids)) if pre else tok.decode(out_ids)
                if pre and vis.tools:
                    said = pre[:pre.index("<tool_call>")] + "<tool_call>" + vis.tools[0] + "</tool_call>"
                if MODEL_ROUTING:  # keep the tools declared; the model may call again or answer
                    convo = convo + [{"role": "assistant", "content": tok.decode(out_ids)},
                                     {"role": "tool", "content": "\n".join(results)}]
                    continue
                convo = [m for m in convo if not (m.get("role") == "system" and "You can call tools" in str(m.get("content")))]
                if extra_sys:
                    convo = [{"role": "system", "content": extra_sys}] + convo
                if searched:  # grounded-QA format from SFT (IndicQA-style), not a tool message
                    lang = script_lang(convo) or "en"
                    cw, qw = CTX_WORDS.get(lang, CTX_WORDS["en"])
                    users = [i for i, m in enumerate(convo) if m.get("role") == "user"]
                    q = convo[users[-1]]["content"]
                    convo = convo[:users[-1]] + [{"role": "user", "content":
                                                   f"{cw}:\n" + "\n\n".join(results) + f"\n\n{qw}: {q}"}]
                else:
                    convo = convo + [{"role": "assistant", "content": said},
                                     {"role": "tool", "content": "\n".join(results)}]
        ms = round((time.time() - t) * 1000)
        yield "end", {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out,
                      "telemetry": {"kannada_tokens": n_out, "kannada_latency_ms": ms, "total_latency_ms": ms}}

    if body.get("stream"):
        def sse():
            yield f"data: {json.dumps(chunk({'role': 'assistant'}))}\n\n"
            finish = "stop"
            for kind, val in run():
                if kind == "client_calls":
                    finish = "tool_calls"
                    yield f"data: {json.dumps(chunk({'tool_calls': openai_calls(val, stream=True)}), ensure_ascii=False)}\n\n"
                    continue
                if kind == "delta":
                    yield f"data: {json.dumps(chunk({'content': val}), ensure_ascii=False)}\n\n"
                elif kind == "tool":
                    yield f"data: {json.dumps(chunk({'tool_call': val}), ensure_ascii=False)}\n\n"
                elif kind == "thinking":  # vendor field like tool_call: {"status": "thinking" | "done", "ms"}
                    yield f"data: {json.dumps(chunk({'thinking': val}))}\n\n"
                else:
                    yield f"data: {json.dumps(chunk({}, finish))}\n\n"
                    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_NAME, 'choices': [], 'usage': val})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    def collect():  # in a worker thread: generation blocks
        text, usage, calls = "", {}, None
        for kind, val in run():
            if kind == "delta":
                text += val
            elif kind == "client_calls":
                calls = openai_calls(val)
            elif kind == "end":
                usage = val
        return text, usage, calls
    text, usage, calls = await run_in_threadpool(collect)
    msg = {"role": "assistant", "content": text.strip() or (None if calls else "")}
    if calls:
        msg["tool_calls"] = calls
    return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
                         "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                         "usage": usage})


if __name__ == "__main__":
    import uvicorn
    # behind a TLS proxy (GCP deployment, 2026-10-22): VAJRAM_HOST=127.0.0.1
    uvicorn.run(app, host=os.environ.get("VAJRAM_HOST", "0.0.0.0"), port=PORT, log_level="warning")
