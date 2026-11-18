"""Local preview server with correct MIME types (Windows maps .js to text/plain). python web/serve.py [port]"""
import http.server, sys, functools, os
H = http.server.SimpleHTTPRequestHandler
H.extensions_map.update({".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".glb": "model/gltf-binary", ".wasm": "application/wasm"})
H.log_message = lambda *a: None
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
handler = functools.partial(H, directory=os.path.dirname(os.path.abspath(__file__)))
http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
