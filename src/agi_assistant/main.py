"""命令行启动入口。"""

from __future__ import annotations

import logging

import uvicorn

from .api import create_app
from .config import load_config


def run() -> None:
    config = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(config), host="0.0.0.0", port=config.server_port)


if __name__ == "__main__":
    run()
