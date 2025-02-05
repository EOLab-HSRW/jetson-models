from http.server import SimpleHTTPRequestHandler, HTTPServer

class CustomHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        if self.path.endswith(".js"):
            self.send_header("Content-Type", "application/javascript")
        super().end_headers()

PORT = 8000
server_address = ("", PORT)
httpd = HTTPServer(server_address, CustomHandler)

print(f"Serving on http://localhost:{PORT}")
httpd.serve_forever()

