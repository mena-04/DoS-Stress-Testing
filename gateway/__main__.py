"""Run the gateway.

    python -m gateway --config configs/ratelimit_queue.yaml --run-id spike-01

Launchable from a Colab cell with subprocess.Popen exactly like vLLM itself,
since the generator, the gateway and the backend all share one runtime.
"""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="DoS mitigation gateway")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--mode", choices=("off", "ratelimit", "ratelimit_queue", "full"))
    parser.add_argument("--upstream", help="vLLM base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--run-id", help="run identifier; names the output directory")
    parser.add_argument("--log-dir")
    parser.add_argument("--log-features", action="store_true", default=None)
    args = parser.parse_args()

    config = load_config(
        args.config,
        mode=args.mode,
        host=args.host,
        port=args.port,
        **{
            "upstream.base_url": args.upstream,
            "logging.run_id": args.run_id,
            "logging.dir": args.log_dir,
            "logging.log_features": args.log_features,
        },
    )

    print(f"gateway mode={config.mode} upstream={config.upstream.base_url}")
    print(f"listening on http://{config.host}:{config.port}")
    print(f"writing {config.logging.dir}/{config.logging.run_id}/")

    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
