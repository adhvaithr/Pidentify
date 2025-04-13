from parser import Parser
from logger import create_logger
from dataset import Dataset
import os
import sys


if __name__ == "__main__":
    user_input = sys.argv
    datafiles, merge_files, merge_on_keys, delay_write = Parser(user_input).process_arguments()    
    file_status_logger = create_logger("file_status", os.path.join(os.getcwd(), os.path.dirname(os.path.realpath(__file__)), "logs", "dataset_converter_log.txt"))
    file_status_logger.info(f"Command line input: {' '.join(user_input)}")
    Dataset(datafiles, merge_files, merge_on_keys, delay_write, file_status_logger).process_dataset()
    