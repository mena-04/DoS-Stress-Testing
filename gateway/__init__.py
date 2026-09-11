"""Per-client rate limiting and queue-pressure shedding in front of vLLM."""

from .config import GatewayConfig, load_config

__all__ = ["GatewayConfig", "load_config", "create_app"]
__version__ = "0.1.0"


def create_app(*args, **kwargs):
    # Imported lazily so that `python -m gateway.anomaly.train` and the unit
    # tests do not need fastapi installed.
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
