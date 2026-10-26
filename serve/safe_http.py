"""Outbound HTTP for model-chosen URLs (fetch_page, search-result pages) without SSRF.

- only http / https, only ports 80 and 443, no userinfo in the URL;
- the host is resolved once and EVERY address must be public (not private, loopback, link-local, multicast,
  reserved, CGNAT or unspecified, IPv4-mapped IPv6 included); the socket then connects to one of those vetted
  addresses, so a DNS answer that changes between check and connect (rebinding) cannot reach an internal host;
- redirects (at most 3) go through the same checks, because every connection is made by the vetted connector;
- environment proxies are ignored (a proxy would make the address check meaningless);
- response size and time are capped.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request

ALLOWED_PORTS = {80, 443}
MAX_REDIRECTS = 3
UA = "Mozilla/5.0 (compatible; vajram-serve)"


class BlockedURL(Exception):
    pass


def _ip_ok(ip: str) -> bool:
    a = ipaddress.ip_address(ip.split("%", 1)[0])
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped is not None:
        a = a.ipv4_mapped
    if isinstance(a, ipaddress.IPv6Address) and a.sixtofour is not None:
        a = a.sixtofour
    return a.is_global and not (a.is_multicast or a.is_reserved or a.is_loopback or a.is_link_local
                                or a.is_private or a.is_unspecified)


def resolve_public(host: str, port: int):
    """All addresses of host; raises BlockedURL unless every one is public."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise BlockedURL(f"cannot resolve host ({e.__class__.__name__})")
    addrs = []
    for fam, _, _, _, sa in infos:
        if not _ip_ok(sa[0]):
            raise BlockedURL("host resolves to a private or reserved address")
        addrs.append((fam, sa))
    if not addrs:
        raise BlockedURL("host has no address")
    return addrs


def _vetted_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *a, **k):
    host, port = address
    last = None
    for fam, sa in resolve_public(host, port):
        s = socket.socket(fam, socket.SOCK_STREAM)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                s.settimeout(timeout)
            s.connect(sa)
            if not _ip_ok(s.getpeername()[0]):  # belt and braces
                raise BlockedURL("connected to a non-public address")
            return s
        except BlockedURL:
            s.close()
            raise
        except OSError as e:
            last = e
            s.close()
    raise last or OSError("connection failed")


class _HTTPConn(http.client.HTTPConnection):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._create_connection = _vetted_create_connection


class _HTTPSConn(http.client.HTTPSConnection):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._create_connection = _vetted_create_connection


class _HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_HTTPConn, req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_HTTPSConn, req, context=self._context)


class _Redirects(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_url(url: str) -> str:
    p = urllib.parse.urlsplit(str(url))
    if p.scheme not in ("http", "https"):
        raise BlockedURL("only http and https URLs are allowed")
    if not p.hostname:
        raise BlockedURL("URL has no host")
    if p.username or p.password:
        raise BlockedURL("URLs with credentials are not allowed")
    port = p.port or (443 if p.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise BlockedURL("only ports 80 and 443 are allowed")
    return url


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _HTTPHandler(), _HTTPSHandler(), _Redirects())
# build_opener keeps the default handlers it is not given replacements for; make sure no plain handler remains
_OPENER.handlers = [h for h in _OPENER.handlers
                    if type(h) not in (urllib.request.HTTPHandler, urllib.request.HTTPSHandler,
                                       urllib.request.HTTPRedirectHandler, urllib.request.FTPHandler,
                                       urllib.request.FileHandler, urllib.request.DataHandler)]


def fetch(url: str, timeout: float = 10, max_bytes: int = 800_000) -> str:
    """Body of a public http(s) URL as text. Raises BlockedURL for refused URLs, other exceptions for network errors."""
    check_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with _OPENER.open(req, timeout=timeout) as r:
        return r.read(max_bytes).decode("utf-8", "ignore")
