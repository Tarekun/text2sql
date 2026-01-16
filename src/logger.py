import sys
from loguru import logger
from src.config import Config


_initialized = False


def configure_logger(config: Config):
    global _initialized

    if not _initialized:
        # log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        # remove default handler (optional)
        logger.remove()

        logger.add(
            sys.stderr,
            level=config.log_level,
            # format=log_format,
            colorize=True,  # Enable colors in terminal
            diagnose=True,
        )
        logger.add(
            "logs/out.log",
            rotation="500 MB",
            # retention="10 days",
            level=config.log_level,
            # format=log_format,
            # compression="zip",  # Compress rotated files
            # enqueue=True,  # Async-safe (for multi-threading)
            # backtrace=True,  # Include error traces
            diagnose=True,  # Include variable values in errors
        )

        _initialized = True
    else:
        raise ValueError(
            "Logger module was already initialized, not allowed to reconfigure it"
        )
