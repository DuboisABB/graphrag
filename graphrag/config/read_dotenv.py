# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing the read_dotenv utility."""

import logging
import os
from pathlib import Path

from dotenv import dotenv_values

log = logging.getLogger(__name__)


def read_dotenv(root: str) -> None:
    """Read a .env file in the given root path."""
    env_path = Path(root) / ".env"
    print(f"Trying to read env from file: {env_path}")
    if env_path.exists():
        log.info("Loading pipeline .env file: %s", env_path)
        env_config = dotenv_values(f"{env_path}")
        for key, value in env_config.items():
            redacted = "<redacted>" if key.upper() in {"API_KEY", "OPENAI_API_KEY"} else value
            log.info("Loaded env var: %s=%s", key, redacted)            
            if key not in os.environ:
                os.environ[key] = value or ""
    else:
        log.info("No .env file found at %s", root)
