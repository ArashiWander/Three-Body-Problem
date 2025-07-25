import http.server
import socketserver
import threading
import os

from threebody.nasa import download_ephemeris


def test_download_ephemeris(tmp_path):
    data = b"abc123"
    source = tmp_path / "source.bsp"
    source.write_bytes(data)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    cwd = os.getcwd()
    os.chdir(tmp_path)
    httpd = socketserver.TCPServer(("localhost", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://localhost:{port}/source.bsp"
        dest = tmp_path / "out.bsp"
        path = download_ephemeris(url, dest)
        assert path.exists()
        assert path.read_bytes() == data
    finally:
        httpd.shutdown()
        thread.join()
        os.chdir(cwd)
