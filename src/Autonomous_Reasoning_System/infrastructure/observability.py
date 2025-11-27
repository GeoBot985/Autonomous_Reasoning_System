import threading
import time
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import defaultdict

logger = logging.getLogger(__name__)

class Metrics:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Metrics, cls).__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.start_time = time.time()

    def increment(self, name: str, value: int = 1):
        with self._lock:
            self.counters[name] += value

    def set_gauge(self, name: str, value: float):
        with self._lock:
            self.gauges[name] = value

    def record_time(self, name: str, duration: float):
        with self._lock:
            # Keep last 100 timings
            self.histograms[name].append(duration)
            if len(self.histograms[name]) > 100:
                self.histograms[name].pop(0)

    def get_metrics(self):
        with self._lock:
            metrics = {
                "uptime": time.time() - self.start_time,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timings": {k: {"avg": sum(v)/len(v) if v else 0, "count": len(v)} for k, v in self.histograms.items()}
            }
            return metrics

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            metrics = Metrics().get_metrics()
            self.wfile.write(json.dumps(metrics, indent=2).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default HTTP logging to avoid clutter
        pass

class HealthServer(threading.Thread):
    def __init__(self, port=8000):
        super().__init__()
        self.port = port
        self.daemon = True # Auto-kill when main thread exits
        self.httpd = None

    def run(self):
        try:
            self.httpd = HTTPServer(('0.0.0.0', self.port), HealthHandler)
            logger.info(f"🏥 Healthz server listening on port {self.port}")
            self.httpd.serve_forever()
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
