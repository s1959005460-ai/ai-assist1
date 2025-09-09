# monitoring.py
import logging
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False
    # create noop stand-ins
    class _Noop:
        def __getattr__(self, _):
            return lambda *a, **k: None
    Counter = Gauge = Histogram = _Noop

logger = logging.getLogger(__name__)

# Define metrics
if PROM_AVAILABLE:
    ROUND_TIME = Histogram('fedgnn_round_seconds', 'Time spent per federated round')
    CLIENT_TRAIN_SECONDS = Histogram('fedgnn_client_train_seconds', 'Client local training seconds', ['client_id'])
    COMM_BYTES = Counter('fedgnn_comm_bytes_total', 'Total bytes communicated')
    AVG_TRAIN_LOSS = Gauge('fedgnn_avg_train_loss', 'Average training loss per round', ['round'])
else:
    ROUND_TIME = CLIENT_TRAIN_SECONDS = COMM_BYTES = AVG_TRAIN_LOSS = None

def start_prometheus_server(port=8001):
    if PROM_AVAILABLE:
        start_http_server(port)
        logger.info("Prometheus metrics server started on %d", port)
    else:
        logger.info("prometheus_client not available; metrics disabled")
