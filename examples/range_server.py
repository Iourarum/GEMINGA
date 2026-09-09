"""A tiny HTTP server that honours Range requests and counts bytes served.

Used to test geminga's HTTP path locally (the Hub is not reachable from every sandbox).
Run:  python examples/range_server.py /path/to/dir 8765
"""
import os, sys, threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

SERVED = {"bytes": 0, "requests": 0, "ranges": 0}

class RangeHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_HEAD(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404); return
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        SERVED["requests"] += 1

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404); return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        SERVED["requests"] += 1
        if rng and rng.startswith("bytes="):
            SERVED["ranges"] += 1
            start, end = rng[6:].split("-", 1)
            if start == "":
                length = int(end); start = max(0, size - length); end = size - 1
            else:
                start = int(start); end = int(end) if end else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers(); return
            n = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(n))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start); self.wfile.write(f.read(n))
            SERVED["bytes"] += n
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            with open(path, "rb") as f:
                self.wfile.write(f.read())
            SERVED["bytes"] += size

def serve(directory: str, port: int) -> HTTPServer:
    os.chdir(directory)
    httpd = HTTPServer(("127.0.0.1", port), RangeHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    return httpd

if __name__ == "__main__":
    httpd = serve(sys.argv[1], int(sys.argv[2]))
    print(f"serving {sys.argv[1]} on http://127.0.0.1:{sys.argv[2]}")
    threading.Event().wait()
