"""Server-side tools for the Vajram demo: calculator (safe AST evaluation) and web search (DuckDuckGo, Wikipedia fallback).

Schemas use the function-calling format Vajram was trained on (glaive / Dolci tool-use rows); the model replies with
<|act_tool|><tool_call>{"name": ..., "arguments": {...}}</tool_call>, the server runs the tool and sends the result
back as a `tool` message.
"""
import ast, json, math, operator, os, re, sys, time, urllib.error, urllib.parse, urllib.request

# verified verse store (Gita + principal Upanishads); see scripture.py
_DEFAULT_SCRIPTURE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripture"))
sys.path.insert(0, os.environ.get("SCRIPTURE_DIR", _DEFAULT_SCRIPTURE))
import scripture  # noqa: E402

try:
    from . import safe_python, safe_http
except ImportError:
    import safe_python, safe_http

TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly (use for any calculation: percentages, GST, interest, "
                       "unit conversions). Supports + - * / ** %, parentheses, sqrt, round, abs, min, max, pi.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "a single arithmetic expression with numbers only; write a percentage as a decimal factor: 15% off 800 = 800 * 0.85, plus 18% GST on 500 = 500 * 1.18"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current or factual information (news, weather, prices, people, places, events, "
                       "dates, recent facts). Returns titles, snippets and links.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search query, in the user's language or English"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "scripture_lookup",
        "description": "Exact text of a verse, shloka or mantra from a verified scripture collection (Bhagavad Gita, "
                       "Isha, Kena, Katha, Prashna, Mundaka, Mandukya Upanishads). Always use this instead of quoting a "
                       "verse from memory. The verse is shown to the user verbatim; you explain its meaning.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "scripture name, e.g. Bhagavad Gita, Katha Upanishad"},
            "ref": {"type": "string", "description": "verse reference, e.g. 2.47 (chapter.verse) or 1.2.3; 'shanti' for the peace invocation"}},
            "required": ["text", "ref"]}}},
]

R2_TOOLS = [  # identical to vajram_train/gen_r2.py BUILTIN (minus translate: no translation backend in the demo)
    {"type": "function", "function": {"name": "python", "description":
        "Run Python code and return what it prints. Use for any calculation (arithmetic, percentages, GST, interest, "
        "EMI, dates, unit conversion, statistics) so the numbers are exact.", "parameters": {"type": "object",
        "properties": {"code": {"type": "string", "description": "Python code; print the result"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "web_search", "description":
        "Search the web for current or factual information (news, weather, prices, people, places, events, dates). "
        "Returns titles, snippets and URLs.", "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "fetch_page", "description": "Read the main text of a web page.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "scripture_lookup", "description":
        "Exact text of a verse, shloka or mantra from the verified scripture collection (Bhagavad Gita and principal "
        "Upanishads). Always use this instead of quoting a verse from memory. The verse is shown to the user verbatim; "
        "you explain its meaning.", "parameters": {"type": "object", "properties": {
        "text": {"type": "string", "description": "scripture name"}, "ref": {"type": "string", "description":
        "verse reference, e.g. 2.47 or 1.2.3; 'shanti' for the peace invocation"}}, "required": ["text", "ref"]}}},
]
_PY_BLOCK = re.compile(r"\b(shutil|subprocess|socket|requests|urllib|ctypes|os\.(remove|unlink|rmdir|system|popen)|"
                       r"__import__|open\(|eval\(|exec\()")


def run_python(code):
    """Evaluate python code.
    By default, uses a restricted AST evaluator (safe_python) with strict step, time,
    and memory limits, without network, filesystem, or arbitrary module access.
    Arbitrary subprocess execution is enabled only if VAJRAM_UNSAFE_PYTHON=1 is set.
    """
    code = str(code or "")
    if os.environ.get("VAJRAM_UNSAFE_PYTHON") != "1":
        return safe_python.run(code)

    # Opt-in unsafe subprocess path
    print("WARNING: Executing arbitrary Python code (VAJRAM_UNSAFE_PYTHON=1). This is dangerous and not sandboxed.",
          file=sys.stderr, flush=True)
    import subprocess, sys as _s, tempfile
    if _PY_BLOCK.search(code):
        return {"error": f"blocked in the demo sandbox: {_PY_BLOCK.search(code).group(0)}"}
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "main.py"), "w", encoding="utf-8").write(code)
        try:
            p = subprocess.run([_s.executable, "main.py"], cwd=d, capture_output=True, text=True, timeout=15,
                               encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            return {"output": (p.stdout + p.stderr).strip()[-2000:] or f"(no output, exit code {p.returncode})"}
        except subprocess.TimeoutExpired:
            return {"error": "timed out after 15 s"}


def fetch_page(url):
    """Fetch the main text of a web page using an SSRF-safe HTTP client."""
    url = str(url or "")
    try:
        html = safe_http.fetch(url, timeout=10, max_bytes=800_000)
    except safe_http.BlockedURL as e:
        return {"error": f"URL blocked: {e}"}
    except Exception as e:
        return {"error": f"could not fetch the page ({e.__class__.__name__})"}
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    html = re.sub(r"(?is)<(script|style|nav|footer|header|aside|table)[^>]*>.*?</\1>", " ", html)
    paras = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", p)).strip() for p in re.findall(r"(?is)<p[^>]*>(.*?)</p>", html)]
    text = "\n".join(re.sub(r"\[\d+\]", "", p) for p in paras if len(p) > 50)
    head = re.sub(r"\s+", " ", title.group(1)).strip() if title else url
    return {"page": f"{head}\n\n{text[:3500]}"} if text else {"error": "the page has no readable text"}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
        ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max, "log": math.log, "log10": math.log10,
          "sin": math.sin, "cos": math.cos, "tan": math.tan, "ceil": math.ceil, "floor": math.floor}
_NAMES = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    raise ValueError("unsupported expression")


def calculator(expression):
    expr = str(expression).replace("×", "*").replace("÷", "/").replace(",", "").replace("^", "**").replace("₹", "")
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)  # 18% -> (18/100)
    expr = re.sub(r"(?i)\s+of\s+", " * ", expr)  # 18% of 500 -> (18/100) * 500
    try:
        v = _eval(ast.parse(expr, mode="eval"))
        if isinstance(v, float):
            v = round(v, 10)
            if v.is_integer():
                v = int(v)
        return {"expression": expression, "result": v}
    except Exception as e:
        return {"expression": expression, "error": f"could not evaluate: {e}"}


WIKI_UA = {"User-Agent": "VajramDemo/0.2 (local chat demo; Python urllib)"}


def _wiki_get(url):
    for attempt in range(2):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=WIKI_UA), timeout=10))
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt:
                raise
            time.sleep(1.5)


def _wikipedia(query, lang):
    q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3})
    hits = _wiki_get(f"https://{lang}.wikipedia.org/w/api.php?{q}")["query"]["search"]
    return [{"title": h["title"], "snippet": re.sub(r"<[^>]+>", "", h["snippet"]),
             "url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(h['title'].replace(' ', '_'))}"} for h in hits]


def _wiki_summary(hit):
    host = hit["url"].split("/wiki/")[0]
    d = _wiki_get(f"{host}/api/rest_v1/page/summary/{urllib.parse.quote(hit['title'].replace(' ', '_'))}")
    return d.get("title", hit["title"]), d.get("extract", "")


def _wiki_for_search(query):
    """The Wikipedia article a search is about, if there is one: the top hit counts only when its title is named in the
    query (all title words, or the title itself, appear in it). Kannada Wikipedia for Kannada script, else English."""
    q = str(query)
    lang = "kn" if re.search("[\u0c80-\u0cff]", q) else "en"
    try:
        hits = _wikipedia(q, lang) or (_wikipedia(q, "en") if lang == "kn" else [])
        if not hits:
            return None
        title = re.sub(r"\s*\(.*?\)", "", hits[0]["title"]).strip()
        tw = [w for w in re.findall(r"\w+", title.lower()) if w not in _STOP]
        qw = set(re.findall(r"\w+", q.lower()))
        if not (title.lower() in q.lower() or (tw and all(w in qw for w in tw))):
            return None
        t, summary = _wiki_summary(hits[0])
        return {"title": f"Wikipedia: {t}", "snippet": summary[:700], "url": hits[0]["url"]} if summary else None
    except Exception:
        return None


def web_search(query, max_results=5):
    """hard 45 s limit (2026-10-23 eval hang: DuckDuckGo / page reads can block for hours and hold the server's slots)"""
    import concurrent.futures as _cf
    ex = _cf.ThreadPoolExecutor(1)
    try:
        return ex.submit(_web_search_inner, query, max_results).result(timeout=45)
    except _cf.TimeoutError:
        return {"query": query, "error": "search timed out"}
    finally:
        ex.shutdown(wait=False)


def _web_search_inner(query, max_results=5):
    """One search tool (2026-10-04: the separate wikipedia tool was merged in; Vajram's training data only has
    web_search). DuckDuckGo + the top pages' matching paragraphs, and the Wikipedia summary first when the query names
    an article."""
    import concurrent.futures as _cf
    wiki = _cf.ThreadPoolExecutor(1).submit(_wiki_for_search, query)
    results = []
    try:
        from ddgs import DDGS
        results = [{"title": r["title"], "snippet": r["body"][:300], "url": r["href"]}
                   for r in DDGS().text(str(query), max_results=max_results)]
    except Exception:
        pass
    if not results:  # DuckDuckGo often answers empty once; one retry
        time.sleep(1)
        try:
            results = [{"title": r["title"], "snippet": r["body"][:300], "url": r["href"]}
                       for r in DDGS().text(str(query), max_results=max_results)]
        except Exception:
            pass
    if not results and not wiki.result():  # fallback: Wikipedia search (Kannada for Kannada script, else English)
        try:
            lang = "kn" if re.search("[ಀ-೿]", str(query)) else "en"
            results = _wikipedia(str(query), lang)
        except Exception as e:
            return {"query": query, "error": f"search failed: {e}"}
    # Snippets are 1-2 cut-off sentences that rarely hold the answer: read the top pages and keep the paragraphs
    # that match the query. Long percent-encoded URLs derail the 7B (probe_toolresult.py), so the model gets
    # title + text only.
    for r in results[:2]:
        text = _page_paragraphs(r.get("url", ""), query)
        if text:
            r["snippet"] = text
    w = wiki.result()
    if w:
        results = [w] + [r for r in results if r.get("url") != w["url"]]
    return {"query": query, "results": [{"title": r["title"], "snippet": r["snippet"][:700], "url": r.get("url", "")}
                                       for r in results[:3]]}


_STOP = set("the a an of in on at to for is are was were who what when where which how did does do and or most "
            "recent latest today now current".split())


def _page_paragraphs(url, query, limit=700):
    """Paragraphs of a web page that share the most words with the query (plain HTML, no JS)."""
    if not url.startswith("http"):
        return ""
    try:
        html = safe_http.fetch(url, timeout=6, max_bytes=600_000)
    except Exception:
        return ""
    html = re.sub(r"(?is)<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>", " ", html)
    paras = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", p)).strip() for p in re.findall(r"(?is)<p[^>]*>(.*?)</p>", html)]
    paras = [re.sub(r"\[\d+\]", "", p) for p in paras if len(p) > 60]
    words = {w for w in re.findall(r"\w+", str(query).lower()) if w not in _STOP and len(w) > 2}
    if not paras or not words:
        return ""
    ranked = sorted(paras, key=lambda p: -sum(w in p.lower() for w in words))
    out = ""
    for p in ranked[:3]:
        if len(out) + len(p) > limit:
            p = p[: max(0, limit - len(out))]
        out += (" " if out else "") + p
        if len(out) >= limit:
            break
    return out.strip()


STD_TOOLS = [  # standard chat tools (no API keys): declared with TOOLS, routed by server.py's typed decision
    {"type": "function", "function": {"name": "weather", "description":
        "Current weather and today's minimum / maximum temperature for a city.", "parameters": {"type": "object",
        "properties": {"city": {"type": "string", "description": "city name in English, e.g. Bengaluru, Mysuru"}},
        "required": ["city"]}}},
    {"type": "function", "function": {"name": "currency_convert", "description":
        "Convert an amount between currencies at today's exchange rate (European Central Bank reference rates).",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "from_currency": {"type": "string", "description": "ISO code, e.g. USD"},
            "to_currency": {"type": "string", "description": "ISO code, e.g. INR"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "get_datetime", "description":
        "Current date, day of the week and time in a timezone (default India).", "parameters": {"type": "object",
        "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Kolkata, Europe/London"}}}}},
    {"type": "function", "function": {"name": "fetch_page", "description": "Read the main text of a web page the user linked.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]
TOOLS += STD_TOOLS
# Pure model routing: every tool the server can run, declared once; the model alone decides whether / which to call.
# python (not calculator) for arithmetic, as in the round-2 training data.
SERVER_TOOLS = R2_TOOLS + [t for t in STD_TOOLS if t["function"]["name"] in ("weather", "currency_convert", "get_datetime")]
_TZ_FALLBACK = {"asia/kolkata": 5.5, "ist": 5.5, "utc": 0, "gmt": 0, "europe/london": 1, "america/new_york": -4,
                "asia/dubai": 4, "asia/singapore": 8, "asia/tokyo": 9, "australia/sydney": 10, "europe/berlin": 2}


def get_datetime(timezone="Asia/Kolkata"):
    import datetime as dt
    tz_name = str(timezone or "Asia/Kolkata").strip() or "Asia/Kolkata"
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo(tz_name))
    except Exception:  # Windows without tzdata: fixed offsets for common zones
        off = _TZ_FALLBACK.get(tz_name.lower())
        if off is None:
            tz_name, off = "Asia/Kolkata", 5.5
        now = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=off)
    return {"timezone": tz_name, "datetime": now.strftime("%A, %d %B %Y, %I:%M %p")}


def currency_convert(amount, from_currency, to_currency):
    try:
        amount = float(str(amount).replace(",", ""))
        f, t = str(from_currency).upper().strip(), str(to_currency).upper().strip()
        q = urllib.parse.urlencode({"amount": amount, "from": f, "to": t})
        d = json.load(urllib.request.urlopen(urllib.request.Request(
            f"https://api.frankfurter.app/latest?{q}", headers={"User-Agent": "vajram-demo/0.1"}), timeout=10))
        val = d["rates"][t]
        return {"amount": amount, "from": f, "to": t, "result": round(val, 2), "rate": round(val / amount, 4) if amount else None,
                "date": d.get("date"), "source": "European Central Bank via frankfurter.app"}
    except Exception as e:
        return {"error": f"conversion failed: {e}"}


def wikipedia(query, lang=None):
    lang = lang if lang in ("kn", "en") else ("kn" if re.search("[ಀ-೿]", str(query)) else "en")
    try:
        hits = _wikipedia(str(query), lang) or (_wikipedia(str(query), "en") if lang == "kn" else [])
        if not hits:
            return {"error": "no Wikipedia article found"}
        title = hits[0]["title"]
        host = hits[0]["url"].split("/wiki/")[0]
        d = _wiki_get(f"{host}/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}")
        return {"title": d.get("title", title), "summary": d.get("extract", "")[:1500], "url": hits[0]["url"]}
    except Exception as e:
        return {"error": f"Wikipedia lookup failed: {e}"}


def weather(query, city=None):
    """Current weather from wttr.in (no API key). City from the argument, else from the query (default Bengaluru)."""
    m = re.search(r"(?i)\b(?:in|at|for)\s+([A-Za-z][A-Za-z .-]+?)(?:\s+today|\s+now|\?|$)", str(query))
    city = (city or (m.group(1).strip() if m else "Bengaluru")).strip().replace(" ", "+")
    try:
        req = urllib.request.Request(f"https://wttr.in/{urllib.parse.quote(city)}?format=j1",
                                     headers={"User-Agent": "curl/8"})
        d = json.load(urllib.request.urlopen(req, timeout=8))
        c, day = d["current_condition"][0], d["weather"][0]
        text = (f"Current weather in {city.replace('+', ' ')}: {c['temp_C']}°C (feels like {c['FeelsLikeC']}°C), "
                f"{c['weatherDesc'][0]['value']}, humidity {c['humidity']}%, wind {c['windspeedKmph']} km/h. "
                f"Today: min {day['mintempC']}°C, max {day['maxtempC']}°C.")
        return {"query": query, "results": [{"title": f"Weather in {city.replace('+', ' ')} (wttr.in)", "snippet": text}]}
    except Exception:
        return None


def run_tool(name, arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"expression": arguments} if name == "calculator" else {"query": arguments}
    arguments = arguments or {}
    if name == "calculator":
        return calculator(arguments.get("expression", ""))
    if name == "web_search":
        return web_search(arguments.get("query", ""))
    if name == "scripture_lookup":
        return scripture.lookup(arguments.get("text", ""), arguments.get("ref", ""))
    if name == "python":
        return run_python(arguments.get("code", ""))
    if name == "fetch_page":
        return fetch_page(arguments.get("url", ""))
    if name == "weather":
        return weather("", city=arguments.get("city") or "Bengaluru") or {"error": "weather service unavailable"}
    if name == "currency_convert":
        return currency_convert(arguments.get("amount", 1), arguments.get("from_currency", "USD"),
                                arguments.get("to_currency", "INR"))
    if name == "get_datetime":
        return get_datetime(arguments.get("timezone"))
    if name == "wikipedia":  # merged into web_search (no longer declared); kept so old clients / prompts still work
        return web_search(arguments.get("query", ""))
    return {"error": f"unknown tool {name}"}


def system_prompt(extra="", r2=False, names=None, tool_list=None):
    """Tool declaration in the exact form used by Vajram's tool-use SFT data (r2: the round-2 tool set; names: only
    these tools; tool_list: exactly these schemas)."""
    ts = tool_list if tool_list is not None else \
        R2_TOOLS if r2 else [t for t in TOOLS if names is None or t["function"]["name"] in names]
    s = "You can call tools. Available tools (JSON schema):\n" + json.dumps(ts, ensure_ascii=False)
    return s + ("\n\n" + extra if extra else "")


def parse_call(text):
    """Tool name + arguments from a possibly malformed <tool_call> body (the 7B sometimes drops a quote or brace)."""
    import json as _j
    try:
        c = _j.loads(text)
        args = c.get("arguments", {})
        if isinstance(args, str):
            args = _j.loads(args)
        return c.get("name", ""), args
    except Exception:
        pass
    m = re.search(r'"name"\s*:\s*"(\w+)"', text)
    vs = re.findall(r'"(expression|query|text|ref|city|url|timezone|from_currency|to_currency|lang)"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    amt = re.search(r'"amount"\s*:\s*"?([\d.,]+)', text)
    if amt:
        vs.append(("amount", amt.group(1)))
    if not m or not vs:
        return "", {}
    return m.group(1), {k: v.rstrip("}").strip() for k, v in vs}


def format_result(name, res, r2=False):
    """Tool result as the model reads it. Search: numbered plain text (JSON made the 7B hallucinate and loop).
    r2: the exact round-2 training formats (gen_r2.py builtin_tool)."""
    import json as _j
    if r2:
        if res.get("error"):
            return "error: " + str(res["error"])
        if name == "web_search":
            rs = res.get("results") or []
            return "\n".join(f"[{i + 1}] {r['title']}\n{r.get('url', '')}\n{r['snippet']}" for i, r in enumerate(rs)) \
                or "No results found."
        if name == "python":  # an empty tool turn teaches nothing; training rows say the same
            return str(res.get("output", "")) or "(no output)"
        if name == "fetch_page":
            return str(res.get("page", ""))
    if name == "web_search" and res.get("results"):
        return "Search results for: " + str(res.get("query", "")) + "\n" + "\n".join(
            f"[{i + 1}] {r['title']}: {r['snippet']}" for i, r in enumerate(res["results"]))
    if res.get("error"):
        return f"{name} failed: {res['error']}"
    if name == "weather" and res.get("results"):
        return res["results"][0]["snippet"]
    if name == "currency_convert":
        return (f"{res['amount']:,.2f} {res['from']} = {res['result']:,.2f} {res['to']} "
                f"(rate {res['rate']}, {res['source']}, {res['date']})")
    if name == "get_datetime":
        return f"Current date and time ({res['timezone']}): {res['datetime']}"
    if name == "wikipedia":
        return f"Wikipedia: {res['title']}\n{res['summary']}"
    if name == "fetch_page" and res.get("page"):  # the 7B degenerates on 3.5k chars of page text: first ~1.2k
        head, _, body = res["page"].partition("\n\n")
        return f"Web page: {head}\n" + body[:1200]
    if name == "scripture_lookup":
        if not res.get("found"):
            return (f"Not found: {res.get('error', '')}. Tell the user the verse is not in the verified collection; "
                    "do not quote it from memory.")
        return (f"{res['text']} {res['ref']} (shown to the user):\n{res['sanskrit']}"
                + (f"\nWord meanings: {res['word_meanings']}" if res.get("word_meanings") else ""))
    return _j.dumps(res, ensure_ascii=False)


def verse_block(res, lang):
    """The verse exactly as stored, for direct display (never retyped by the model): Kannada script for Kannada
    users, Devanagari otherwise, then IAST and the source line."""
    main = res["kannada_script"] if lang == "kn" else res["sanskrit"]
    br = lambda x: "  \n".join(l.strip() for l in x.split("\n"))  # markdown hard line breaks
    iast = "  \n".join(f"_{l.strip()}_" for l in res["iast"].split("\n") if l.strip())
    return (f"**{res['text']} {res['ref']}**\n\n{br(main)}\n\n{iast}\n\n"
            f"<sub>Source: {res['source']} — {res['license']}</sub>\n\n")
