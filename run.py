#!/usr/bin/env python3
"""
run.py  —  NSE Analyser live chart proxy server
  python3 run.py          → port 8000, Kite mode
  python3 run.py --yahoo  → port 8000, Yahoo only  
  python3 run.py 8001     → port 8001 (called by Streamlit)
"""
import http.server, urllib.request, urllib.parse, json, os, sys, time

PORT     = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
USE_KITE = "--yahoo" not in sys.argv

# Always-safe fallbacks so /api/status never crashes
LAST_PRICE      = {}
INSTRUMENTS     = {}
TOKEN_BY_SYMBOL = {}
get_candles     = None
get_last_price  = None
_kite_ready     = False
_feed           = None

if USE_KITE:
    try:
        from kite_auth import load_token
        from kite_feed import (start as kite_start, get_candles, get_last_price,
                                INSTRUMENTS, TOKEN_BY_SYMBOL, LAST_PRICE)
        token = load_token()
        _feed = kite_start(token)
        time.sleep(2)
        _kite_ready = True
        print(f"  Kite feed active — {len(INSTRUMENTS)} instruments")
    except Exception as e:
        print(f"  Kite unavailable ({e}) — Yahoo fallback")


def _sym(query: str, default: str) -> str:
    """Extract sym= from raw query string, preserving '=' in CL=F, NG=F etc."""
    for part in query.split('&'):
        if part.lower().startswith('sym='):
            return urllib.parse.unquote(part[4:])
    return default


class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args): pass

    def do_GET(self):
        p    = urllib.parse.urlparse(self.path)
        path = p.path
        sym  = _sym(p.query, '')

        # ── index → live_chart.html ────────────────────────────────────────
        if path in ('/', '', '/index.html', '/live_chart.html'):
            fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'live_chart.html')
            try:
                self._send(200, 'text/html; charset=utf-8',
                           open(fpath, 'rb').read())
            except Exception as e:
                self._json(500, {"error": str(e)})
            return

        # ── /api/status ────────────────────────────────────────────────────
        if path == '/api/status':
            self._json(200, {"kite_active": _kite_ready,
                             "ticks": len(LAST_PRICE)})
            return

        # ── /api/watchlist ─────────────────────────────────────────────────
        if path == '/api/watchlist':
            self._json(200, [{"token": t, "symbol": s}
                             for t, s in INSTRUMENTS.items()])
            return

        # ── /api/price  (lightweight per-second tick) ──────────────────────
        if path == '/api/price':
            sym = sym.upper() if sym else 'BEL'
            if _kite_ready:
                token = TOKEN_BY_SYMBOL.get(sym)
                if token:
                    from kite_feed import _current_bar
                    self._json(200, {
                        "price":    get_last_price(token),
                        "bar_time": _current_bar.get(token, {}).get('time'),
                        "source":   "kite"
                    })
                    return
            yf = sym if (sym.endswith('=F') or sym.startswith('^')) else sym + '.NS'
            try:
                meta = self._ymeta(yf)
                self._json(200, {"price": meta.get('regularMarketPrice'),
                                 "bar_time": None, "source": "yahoo"})
            except Exception as e:
                self._json(502, {"error": str(e)})
            return

        # ── /api/chart  (full candle data) ─────────────────────────────────
        if path == '/api/chart':
            if not sym:
                sym = 'CL=F'
            su = sym.upper()
            # Kite for NSE stocks
            if _kite_ready and not (sym.endswith('=F') or sym.startswith('^')):
                token = TOKEN_BY_SYMBOL.get(su)
                if token:
                    candles = get_candles(token)
                    if candles:
                        self._json(200, {
                            "source":     "kite",
                            "symbol":     su,
                            "token":      token,
                            "candles":    candles,
                            "last_price": get_last_price(token),
                        })
                        return
            # Yahoo fallback (always for commodities)
            yf = sym if (sym.endswith('=F') or sym.startswith('^')) else su + '.NS'
            self._yproxy(yf)
            return

        self._json(404, {"error": f"not found: {path}"})

    def _send(self, code: int, ct: str, body: bytes):
        self.send_response(code)
        self.send_header('Content-Type', ct)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data):
        self._send(code, 'application/json', json.dumps(data).encode())

    def _ymeta(self, sym: str) -> dict:
        enc = urllib.parse.quote(sym, safe='')
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}"
               f"?range=1d&interval=1m&includePrePost=false")
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.loads(r.read())['chart']['result'][0]['meta']

    def _yproxy(self, sym: str):
        enc  = urllib.parse.quote(sym, safe='')
        last = 'no response'
        for host in ['query1', 'query2']:
            url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/{enc}"
                   f"?range=1d&interval=1m&includePrePost=false")
            try:
                req = urllib.request.Request(
                    url, headers={'User-Agent': 'Mozilla/5.0',
                                  'Accept': 'application/json'})
                with urllib.request.urlopen(req, timeout=8) as r:
                    self._send(200, 'application/json', r.read())
                return
            except Exception as e:
                last = str(e)
        self._json(502, {"error": last})


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    mode = "Kite WebSocket" if _kite_ready else "Yahoo Finance"
    print(f"\n  Live Chart Server ({mode}) → http://localhost:{PORT}\n")
    try:
        with http.server.HTTPServer(('', PORT), Handler) as s:
            s.serve_forever()
    except KeyboardInterrupt:
        if _feed:
            try: _feed.stop()
            except: pass
        print("\n  Stopped.")
