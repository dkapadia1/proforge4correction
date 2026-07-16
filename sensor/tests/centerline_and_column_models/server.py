from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import os
import sys

# ensure local module import works when server is run from this file's directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import analyze_six


def parse_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value, default=False):
    if value is None:
        return default
    value = value.strip().lower()
    return value in ('1', 'true', 'yes', 'on', 'y')


class MaxDeviationHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != '/max_deviation':
            self._send_json(404, {'error': 'Endpoint not found', 'path': parsed.path})
            return

        params = parse_qs(parsed.query)
        image_path = params.get('path', [None])[0]
        if not image_path:
            self._send_json(400, {'error': 'Missing required query parameter: path'})
            return

        # support relative paths from server directory
        if not os.path.isabs(image_path):
            image_path = os.path.join(ROOT_DIR, image_path)

        med_ksize = parse_int(params.get('med_ksize', [3])[0], 3)
        search_radius = parse_int(params.get('search_radius', [4])[0], 4)
        smooth_method = params.get('smooth_method', ['savgol'])[0]
        savgol_window = parse_int(params.get('savgol_window', [7])[0], 7)
        visualize = parse_bool(params.get('visualize', ['false'])[0], False)

        try:
            max_dev = analyze_six.max_deviation_from_centerline(
                image_path,
                med_ksize=med_ksize,
                search_radius=search_radius,
                smooth_method=smooth_method,
                savgol_window=savgol_window,
                visualize=visualize,
            )
        except FileNotFoundError as exc:
            self._send_json(404, {'error': str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {'error': 'Internal server error', 'details': str(exc)})
            return

        self._send_json(200, {'max_deviation': float(max_dev), 'image_path': image_path})

    def log_message(self, format, *args):
        # silence default logging or override to keep console clean
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))


def run_server(host='0.0.0.0', port=8000):
    server = HTTPServer((host, port), MaxDeviationHandler)
    print(f"Serving max deviation endpoint at http://{host}:{port}/max_deviation")
    print("Use query parameter 'path' to specify the image path.")
    server.serve_forever()


if __name__ == '__main__':
    run_server()
