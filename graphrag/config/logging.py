# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Logging utilities. A unified way for enabling logging."""

import logging
import re
from pathlib import Path

from graphrag.config.enums import ReportingType
from graphrag.config.models.graph_rag_config import GraphRagConfig


def enable_logging(log_filepath: str | Path, verbose: bool = False) -> None:
    """Enable logging to a file.

    Parameters
    ----------
    log_filepath : str | Path
        The path to the log file.
    verbose : bool, default=False
        Whether to log debug messages.
    """
    log_filepath = Path(log_filepath)
    log_filepath.parent.mkdir(parents=True, exist_ok=True)
    log_filepath.touch(exist_ok=True)

    logging.basicConfig(
        filename=log_filepath,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def enable_logging_with_config(
    config: GraphRagConfig, verbose: bool = False
) -> tuple[bool, str]:
    """Enable logging to a file based on the config.

    Parameters
    ----------
    config : GraphRagConfig
        The configuration.
    timestamp_value : str
        The timestamp value representing the directory to place the log files.
    verbose : bool, default=False
        Whether to log debug messages.

    Returns
    -------
    tuple[bool, str]
        A tuple of a boolean indicating if logging was enabled and the path to the log file.
        (False, "") if logging was not enabled.
        (True, str) if logging was enabled.
    """
    if config.reporting.type == ReportingType.file:
        log_path = Path(config.reporting.base_dir) / "indexing-engine.log"
        enable_logging(log_path, verbose)
        return (True, str(log_path))
    return (False, "")

class AdvancedLineFormatter(logging.Formatter):
    def format(self, record):
        formatted = super().format(record)
        # Convert all escape sequences
        formatted = bytes(formatted, 'utf-8').decode('unicode_escape')
        return formatted

def enable_search_logging_with_config(
    config: GraphRagConfig, verbose: bool = True
) -> tuple[bool, str]:
    if config.reporting.type == "file":
        log_path = Path(config.reporting.base_dir) / "search_debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
        
        search_logger = logging.getLogger("graphrag.search")
        search_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        # Clear any pre-existing handlers so our new one is used.
        search_logger.handlers.clear()
        
        fh = logging.FileHandler(str(log_path))
        fh.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = AdvancedLineFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        search_logger.addHandler(fh)
        return (True, str(log_path))
    return (False, "")