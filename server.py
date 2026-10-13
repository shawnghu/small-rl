#!/usr/bin/env python3
"""Like `python -m http.server`, but directory listings are sorted by
most-recently-modified first instead of alphabetically.

Usage:
    python server.py [PORT] [--bind ADDR] [--directory DIR]
"""
import http.server
import os


class MTimeHandler(http.server.SimpleHTTPRequestHandler):
    def list_directory(self, path):
        # Only the toplevel listing is sorted by mtime; everything below it
        # falls back to the stdlib's alphabetical listing.
        if self.path not in ("/", ""):
            return super().list_directory(path)

        try:
            names = os.listdir(path)
        except OSError:
            return super().list_directory(path)

        def mtime(name):
            try:
                return os.stat(os.path.join(path, name)).st_mtime
            except OSError:
                return 0.0

        names.sort(key=mtime, reverse=True)
        return self._render(path, names)

    def _render(self, path, names):
        # Reimplements SimpleHTTPRequestHandler.list_directory's HTML rendering
        # but skips the alphabetical sort, keeping `names` as given.
        import html
        import io
        import urllib.parse
        from http import HTTPStatus

        try:
            displaypath = urllib.parse.unquote(self.path, errors="surrogatepass")
        except UnicodeDecodeError:
            displaypath = urllib.parse.unquote(self.path)
        displaypath = html.escape(displaypath, quote=False)
        enc = "utf-8"
        title = f"Directory listing for {displaypath}"
        r = []
        r.append("<!DOCTYPE HTML>")
        r.append('<html lang="en">')
        r.append("<head>")
        r.append(f'<meta charset="{enc}">')
        r.append(f"<title>{title}</title>")
        r.append("</head>")
        r.append(f"<body>\n<h1>{title}</h1>")
        r.append("<hr>\n<ul>")
        for name in names:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"
            if os.path.islink(fullname):
                displayname = name + "@"
            r.append(
                '<li><a href="%s">%s</a></li>'
                % (
                    urllib.parse.quote(linkname, errors="surrogatepass"),
                    html.escape(displayname, quote=False),
                )
            )
        r.append("</ul>\n<hr>\n</body>\n</html>\n")
        encoded = "\n".join(r).encode(enc, "surrogateescape")
        f = io.BytesIO()
        f.write(encoded)
        f.seek(0)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", f"text/html; charset={enc}")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        return f


if __name__ == "__main__":
    import argparse
    import contextlib
    import socket

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", "-b", metavar="ADDRESS",
                        help="bind to this address (default: all interfaces)")
    parser.add_argument("--directory", "-d", default=os.getcwd(),
                        help="serve this directory (default: current directory)")
    parser.add_argument("port", default=8000, type=int, nargs="?",
                        help="bind to this port (default: 8000)")
    args = parser.parse_args()

    class DualStackServer(http.server.ThreadingHTTPServer):
        def server_bind(self):
            with contextlib.suppress(Exception):
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            return super().server_bind()

        def finish_request(self, request, client_address):
            self.RequestHandlerClass(request, client_address, self,
                                     directory=args.directory)

    http.server.test(HandlerClass=MTimeHandler, ServerClass=DualStackServer,
                     port=args.port, bind=args.bind)
