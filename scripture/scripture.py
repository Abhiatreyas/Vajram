#!/usr/bin/env python3
"""Verified scripture store + lookup for the `scripture_lookup` tool (SFT spec §9.11).

Principle: exact text travels from a verified source to the user WITHOUT the model retyping it. The tool returns
the verse; the server shows it verbatim; the model only writes the meaning / explanation around it.

Store: scripture/data/store.jsonl, one verse per line:
  {"text_id", "text_name", "ref", "section", "verse", "sanskrit", "word_meanings", "source", "license"}
Kannada script and IAST are derived at lookup time by rule (indic_transliteration), never stored or generated.

  python scripture.py build                 # gita/gita + Upanishads (wikisource_upanishads.jsonl) -> store.jsonl
  python scripture.py lookup gita 2.47
  python scripture.py lookup "isha" 1
"""
import json, os, re, sys, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
STORE = os.path.join(DATA, "store.jsonl")

# canonical id -> names users may type (English, Sanskrit, Kannada); matched after normalisation
TEXTS = {
    "bhagavad_gita": ("Bhagavad Gita", ["bhagavad gita", "bhagavadgita", "gita", "geeta", "bhagavad geeta",
                                        "भगवद्गीता", "श्रीमद्भगवद्गीता", "ಭಗವದ್ಗೀತೆ", "ಗೀತೆ", "ಭಗವದ್ಗೀತಾ"]),
    "isha": ("Isha Upanishad", ["isha", "isa", "ishavasya", "isavasya", "ईशोपनिषत्", "ईशावास्योपनिषद्", "ಈಶಾವಾಸ್ಯ", "ಈಶೋಪನಿಷತ್"]),
    "kena": ("Kena Upanishad", ["kena", "केनोपनिषद्", "ಕೇನೋಪನಿಷತ್", "ಕೇನ"]),
    "katha": ("Katha Upanishad", ["katha", "kathopanishad", "कठोपनिषद्", "ಕಠೋಪನಿಷತ್", "ಕಠ"]),
    "prashna": ("Prashna Upanishad", ["prashna", "prasna", "प्रश्नोपनिषद्", "ಪ್ರಶ್ನೋಪನಿಷತ್"]),
    "mundaka": ("Mundaka Upanishad", ["mundaka", "मुण्डकोपनिषद्", "ಮುಂಡಕೋಪನಿಷತ್", "ಮುಂಡಕ"]),
    "mandukya": ("Mandukya Upanishad", ["mandukya", "माण्डूक्योपनिषत्", "माण्डुक्योपनिषत्", "ಮಾಂಡೂಕ್ಯೋಪನಿಷತ್", "ಮಾಂಡೂಕ್ಯ"]),
    "taittiriya": ("Taittiriya Upanishad", ["taittiriya", "तैत्तिरीयोपनिषद्", "ತೈತ್ತಿರೀಯೋಪನಿಷತ್", "ತೈತ್ತಿರೀಯ"]),
}


def _norm(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(r"[\s\-_.]+", " ", s).strip()
    return re.sub(r"\b(upanishad|upanisad|upanishat|shloka|sloka|verse|chapter)\b", "", s).strip()


ALIASES = {_norm(a): tid for tid, (_, names) in TEXTS.items() for a in names + [tid, TEXTS[tid][0]]}


def resolve_text(name):
    n = _norm(name)
    if n in ALIASES:
        return ALIASES[n]
    # longest alias contained in the query as WHOLE words (2026-10-23: plain substring matching resolved
    # "Hanuman Chalisa" to Isha via "isa" and showed Isha mantra 1 for a Hanuman Chalisa request)
    for a, tid in sorted(ALIASES.items(), key=lambda x: -len(x[0])):
        if a and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", n):
            return tid
    return None


_KN_DIGITS = str.maketrans("೦೧೨೩೪೫೬೭೮೯०१२३४५६७८९", "01234567890123456789")


def parse_ref(ref):
    """'2.47', '2 47', 'chapter 2 verse 47', '೨.೪೭' -> ('2', '47'); '1.2.3' -> ('1.2', '3'); '1' -> (None, '1')."""
    nums = [str(int(n)) for n in re.findall(r"\d+", str(ref).translate(_KN_DIGITS))]
    if len(nums) >= 2:
        return ".".join(nums[:-1]), nums[-1]
    return (None, nums[0]) if nums else (None, None)


_STORE = None


def store():
    global _STORE
    if _STORE is None:
        _STORE = {}
        if os.path.exists(STORE):
            for line in open(STORE, encoding="utf-8"):
                v = json.loads(line)
                _STORE[(v["text_id"], str(v.get("section") or ""), str(v["verse"]))] = v
    return _STORE


def render(v):
    from indic_transliteration import sanscript as s
    sa = v["sanskrit"]
    return {"text": TEXTS[v["text_id"]][0], "ref": v["ref"], "sanskrit": sa,
            "kannada_script": s.transliterate(sa, s.DEVANAGARI, s.KANNADA),
            "iast": s.transliterate(sa, s.DEVANAGARI, s.IAST),
            "word_meanings": v.get("word_meanings") or "", "source": v["source"], "license": v["license"]}


def lookup(text, ref=""):
    tid = resolve_text(text)
    if not tid:
        return {"found": False, "error": f"text not in the verified collection: {text!r}",
                "available": [TEXTS[t][0] for t in TEXTS]}
    section, verse = parse_ref(ref)
    if re.search(r"shant|santi|peace|शान्ति|शांति|ಶಾಂತಿ", str(ref).lower()):  # opening peace invocation
        section, verse = None, "shanti"
    st = store()
    keys = [k for k in st if k[0] == tid]
    if not keys:
        return {"found": False, "error": f"{TEXTS[tid][0]} is listed but not loaded yet"}
    if verse is None:
        return {"found": False, "error": "give a verse reference, e.g. 2.47", "text": TEXTS[tid][0]}
    hit = st.get((tid, section or "", verse)) or (st.get((tid, "", verse)) if section is None else None)
    if hit is None and section is None:  # single number for a sectioned text: first section
        cands = sorted((k for k in keys if k[2] == verse), key=lambda k: int(k[1] or 0))
        hit = st[cands[0]] if cands else None
    if hit is None and section and all(k[1] == "" for k in keys):  # over-specified ref for an unsectioned text
        hit = st.get((tid, "", verse))                               # (Isha "1.1.1" -> mantra 1)
    if hit is None:
        return {"found": False, "error": f"{TEXTS[tid][0]} {ref} not found in the verified collection"}
    return {"found": True, **render(hit)}


# ---------------------------------------------------------------- build
def build():
    rows = []
    for v in json.load(open(os.path.join(DATA, "gita_verse.json"), encoding="utf-8")):  # gita/gita, Unlicense
        sa = re.sub(r"\s*\n\s*\n\s*", "\n", v["text"]).strip().replace("।।", "॥")
        sa = re.sub(r"।(?=[ऀ-ॣ०-ॿ])", "।\n", sa)  # half-lines run together ("च।निर्ममो")
        rows.append({"text_id": "bhagavad_gita", "text_name": "Bhagavad Gita", "section": str(v["chapter_number"]),
                     "verse": str(v["verse_number"]), "ref": f"{v['chapter_number']}.{v['verse_number']}",
                     "sanskrit": sa, "word_meanings": (v.get("word_meanings") or "").strip(),
                     "source": "github.com/gita/gita", "license": "Unlicense (public domain)"})
    up = os.path.join(DATA, "wikisource_upanishads.jsonl")
    if os.path.exists(up):
        rows += [json.loads(l) for l in open(up, encoding="utf-8")]
    with open(STORE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in rows:
        by[r["text_id"]] = by.get(r["text_id"], 0) + 1
    print(f"{len(rows)} verses -> {STORE}: {by}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["lookup"]:
        print(json.dumps(lookup(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""), ensure_ascii=False, indent=1))
