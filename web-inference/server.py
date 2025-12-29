"""
Simple HTTP server for HumChess web inference.

Usage:
    uv run python web-inference/server.py --port 8080
"""

import argparse
import http.server
import socketserver
from pathlib import Path


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS headers and proper MIME types."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self):
        # Add CORS headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        # Required for SharedArrayBuffer (needed for int64 support in WASM)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def guess_type(self, path):
        """Override MIME types for ONNX and WASM files."""
        if path.endswith(".onnx"):
            return "application/octet-stream"
        if path.endswith(".wasm"):
            return "application/wasm"
        return super().guess_type(path)

    def do_OPTIONS(self):
        """Handle preflight requests."""
        self.send_response(200)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="HumChess web inference server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    global STATIC_DIR
    STATIC_DIR = Path(__file__).parent / "static"

    with socketserver.TCPServer((args.host, args.port), CORSHTTPRequestHandler) as httpd:
        print(f"Serving HumChess at http://{args.host}:{args.port}")
        print(f"Static directory: {STATIC_DIR}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
