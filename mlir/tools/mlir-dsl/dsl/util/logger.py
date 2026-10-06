# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
The DSL's process-wide logger.

The package logs through one :class:`logging.Logger`, reached with
:func:`log`; :func:`setup_log` (re)configures it for a DSL prefix from the
``{prefix}_LOG_TO_CONSOLE`` / ``{prefix}_LOG_TO_FILE`` / ``{prefix}_LOG_LEVEL``
settings (see ``LogEnvironmentManager`` in ``core/env_manager.py``). Until a
manager is constructed the logger is the disabled ``"generic"`` one.
"""

import logging

__all__ = ["log", "setup_log"]

logger: logging.Logger

# Above CRITICAL, so no record passes: the level that means "logging is off".
LOG_DISABLED_LEVEL = logging.CRITICAL + 1


def log() -> logging.Logger:
    """Return the logger configured by the last :func:`setup_log` call."""
    return logger


def _resolve_log_level(log_level: int) -> int:
    """Map the ``LOG_LEVEL`` setting to a :mod:`logging` level.

    ``0`` disables logging; any other value (``1`` = everything, or a standard
    ``logging`` level) is used as is.
    """
    if log_level == 0:
        return LOG_DISABLED_LEVEL
    return log_level


def setup_log(
    name: str,
    log_to_console: bool = False,
    log_to_file: bool = False,
    log_file_path: str | None = None,
    log_level: int = 1,
) -> logging.Logger:
    """Configure the DSL logger with console and/or file handlers.

    Replaces the handlers of the ``name`` logger, so calling it again does not
    duplicate output. With neither sink enabled the logger is set to
    :data:`LOG_DISABLED_LEVEL`, whatever ``log_level`` says.

    :param name: Name of the logger to configure (the DSL prefix)
    :type name: str
    :param log_to_console: Whether to log to stderr, defaults to False
    :type log_to_console: bool, optional
    :param log_to_file: Whether to log to a file, defaults to False
    :type log_to_file: bool, optional
    :param log_file_path: Path of the log file; defaults to ``<name>.log`` in
        the working directory when ``log_to_file`` is True and no path is given
    :type log_file_path: str, optional
    :param log_level: Logging verbosity: 0=disabled, 1=all messages
        (debug and above, the default), or a standard ``logging`` level
        (10=debug, 20=info, 30=warning, 40=error, 50=critical)
    :type log_level: int, optional
    :return: The configured logger, also returned by :func:`log` from now on
    :rtype: logging.Logger
    """
    log_level = _resolve_log_level(log_level)
    global logger
    logger = logging.getLogger(name)
    if log_to_console or log_to_file:
        logger.setLevel(log_level)
    else:
        logger.setLevel(LOG_DISABLED_LEVEL)

    # Replace, rather than add to, the handlers of a logger set up before.
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s"
    )

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_to_file:
        file_handler = logging.FileHandler(log_file_path or f"{name}.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_log("generic")
