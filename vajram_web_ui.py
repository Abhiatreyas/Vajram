#!/usr/bin/env python3
"""
Vajram Web UI Launcher.
Opens the built-in chat interface in your default browser.
If the Vajram bridge server is not already running on port 8080, it launches it automatically.
"""

import os
import sys
import time
import urllib.request
import webbrowser
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PORT = int(os.environ.get("VAJRAM_PORT", "8080"))
URL = f"http://127.0.0.1:{PORT}"
HEALTH_URL = f"{URL}/health"


def is_server_running() -> bool:
    try:
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def main():
    print("=" * 60)
    print("✦ Vajram (ವಜ್ರಂ) 8.4B — Indic Web Interface")
    print("=" * 60)

    if is_server_running():
        print(f"[✓] Vajram bridge server detected at {URL}")
    else:
        print(f"[i] Server not detected on port {PORT}. Launching serve/server.py...")
        server_script = os.path.join(os.path.dirname(__file__), "serve", "server.py")
        if not os.path.exists(server_script):
            print(f"[✗] Error: Could not locate {server_script}")
            sys.exit(1)

        env = os.environ.copy()
        proc = subprocess.Popen([sys.executable, server_script], env=env)
        print("[i] Waiting for server initialization...")
        for _ in range(30):
            time.sleep(1)
            if is_server_running():
                print(f"[✓] Server initialized successfully on {URL}!")
                break
        else:
            print("[!] Warning: Server did not respond within 30s. Check logs.")

    print(f"\n[➜] Opening Web Chat UI at {URL} ...")
    webbrowser.open(URL)
    print("\nPress Ctrl+C to exit launcher.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting Vajram Web UI launcher.")


if __name__ == "__main__":
    main()
