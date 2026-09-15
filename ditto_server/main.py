from __future__ import annotations

import uvicorn
from dotenv import load_dotenv

from ditto_server.app import create_app
from ditto_server.config import DittoServiceConfig


load_dotenv()
config = DittoServiceConfig.from_env()
app = create_app(config)


if __name__ == "__main__":
    uvicorn.run(
        "ditto_server.main:app",
        host=config.host,
        port=config.port,
        ssl_certfile=(
            str(config.ssl_certfile.resolve())
            if config.ssl_certfile is not None
            else None
        ),
        ssl_keyfile=(
            str(config.ssl_keyfile.resolve())
            if config.ssl_keyfile is not None
            else None
        ),
    )
