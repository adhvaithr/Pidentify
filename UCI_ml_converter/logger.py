import logging
import sys


def create_logger(logger_name: str) -> logging.Logger:
    """
    Create a Logger object that outputs logs to stderr in the format: "mm/dd/yyyy hh:mm:ss A.M./P.M.: log_message".
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if create_logger is called more than once.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s",
                                               datefmt="%m/%d/%Y %I:%M:%S %p"))
        logger.addHandler(handler)

    return logger
