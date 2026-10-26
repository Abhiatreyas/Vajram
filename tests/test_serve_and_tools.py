"""Unit and integration tests for tools, security boundaries, and server utilities.
"""
import json
import os
import sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SERVE_DIR = os.path.abspath(os.path.join(HERE, "..", "serve"))
SCRIPTURE_DIR = os.path.abspath(os.path.join(HERE, "..", "scripture"))

sys.path.insert(0, SERVE_DIR)
sys.path.insert(0, SCRIPTURE_DIR)
import tools
import server
import safe_http
import safe_python
import scripture


def test_tool_call_parsing():
    # Valid JSON
    raw = json.dumps({"name": "calculator", "arguments": {"expression": "25 * 40"}})
    name, args = tools.parse_call(raw)
    assert name == "calculator"
    assert args == {"expression": "25 * 40"}

    # Malformed / truncated call recovery
    partial = '{"name": "web_search", "arguments": {"query": "weather in Mysuru"'
    name, args = tools.parse_call(partial)
    assert name == "web_search"
    assert args.get("query") == "weather in Mysuru"


def test_fix_tool_open():
    TC_OPEN = server.tok.encode("<tool_call>")[0]
    TC_CLOSE = server.tok.encode("</tool_call>")[0]

    # Model opened with </tool_call> instead of <tool_call>
    corrupted_ids = [TC_CLOSE, 1234, 5678, TC_CLOSE]
    repaired = server.fix_tool_open(corrupted_ids)
    assert repaired[0] == TC_OPEN
    assert repaired[-1] == TC_CLOSE

    # Normal sequence should remain untouched
    normal_ids = [TC_OPEN, 1234, 5678, TC_CLOSE]
    assert server.fix_tool_open(normal_ids) == normal_ids


def test_safe_python_and_calculator():
    # Basic math
    res = tools.calculator("18% of 500 + 45")
    assert res.get("result") == 135

    # Safe python arithmetic
    py_res = safe_python.run("x = 25\ny = 40\nprint(x * y)")
    assert py_res.get("output") == "1000"

    # Math module functions
    math_res = safe_python.run("import math\nprint(round(math.sqrt(144) + math.pi, 2))")
    assert "15.14" in math_res.get("output", "")

    # Security check: imports outside whitelist must be refused
    bad_import = safe_python.run("import os\nos.system('echo test')")
    assert "error" in bad_import
    assert "not allowed" in bad_import["error"]

    # Security check: subprocess / eval blocked
    bad_eval = safe_python.run("eval('2+2')")
    assert "error" in bad_eval


def test_ssrf_fetch_page_refusal():
    # Private / loopback addresses must be refused
    loopback = tools.fetch_page("http://127.0.0.1:8080/secret")
    assert "error" in loopback
    assert "blocked" in loopback["error"].lower() or "not a public" in loopback["error"].lower()

    private_net = tools.fetch_page("http://192.168.1.1/admin")
    assert "error" in private_net
    assert "blocked" in private_net["error"].lower() or "not a public" in private_net["error"].lower()

    cloud_metadata = tools.fetch_page("http://169.254.169.254/latest/meta-data")
    assert "error" in cloud_metadata
    assert "blocked" in cloud_metadata["error"].lower() or "not a public" in cloud_metadata["error"].lower()

    bad_scheme = tools.fetch_page("file:///etc/passwd")
    assert "error" in bad_scheme


def test_scripture_lookup_verbatim():
    # Gita 2.47 lookup
    res = scripture.lookup("bhagavad_gita", "2.47")
    assert res.get("found") is True
    assert "कर्मण्येवाधिकारस्ते" in res.get("sanskrit", "")
    assert res.get("ref") == "2.47"
    assert "Unlicense" in res.get("license", "")

    # Verse formatting helper
    block = tools.verse_block(res, "kn")
    assert "2.47" in block
    assert "ಕರ್ಮಣ್ಯೇವಾಧಿಕಾರಸ್ತೇ" in block  # Kannada script rendered verbatim
    assert "Source:" in block


def test_health_endpoint():
    res = server.health()
    assert "status" in res
    assert "model" in res
    assert res["model"] == "vajram-8b"
