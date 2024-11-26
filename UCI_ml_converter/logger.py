import logging
import os


def create_logger(logger_name: str, log_file_path: str) -> logging.Logger:
    """
    Create a Logger object that outputs logs in the format: "mm/dd/yyyy hh:mm:ss A.M./P.M.: log_message".
    """
    log_file_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)
    logger = logging.getLogger(logger_name)
    logging.basicConfig(filename=log_file_path, encoding="utf-8", format="%(asctime)s: %(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.DEBUG)
    return logger    
