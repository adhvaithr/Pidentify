from dataclasses import dataclass
from datafile import DataFile
import pandas as pd
import logging
import sys


@dataclass
class Dataset:
    """
    Represent entire dataset with a logger for errors.
    """
    datafiles: list[DataFile]
    merge_files: dict[str: list[DataFile]]
    delay_write: bool
    _logger: logging.Logger
    _success: bool = True


    def process_dataset(self) -> None:
        """
        Process all datafiles within the dataset and write the results to the standard output.
        """       
        self._logger.info(f"Processing dataset: {', '.join([datafile.file_path for datafile in self.datafiles])}")
        loaded_dfs = []
        for datafile in self.datafiles:
            try:
                if datafile.file_path in self.merge_files:
                    df = datafile.process_file(*self.merge_files[datafile.file_path], merge=True)
                else:
                    df = datafile.process_file()
            except Exception as e:
                self._logger.error(f"{datafile.file_path}: {type(e)} - {e}")
                self._success = False
                continue   
            if self.delay_write:
                loaded_dfs.append(df)
            else:
                Dataset._output_csv(df)
        if self.delay_write:
            df = pd.concat(loaded_dfs, ignore_index=True)
            Dataset._output_csv(df)
        status = "success" if self._success else "failed"
        self._logger.info(f"Status: {status}\n")


    @staticmethod
    def _output_csv(df: pd.DataFrame) -> None:
        """
        Write contents of dataframe to the standard output.
        """
        keep_col_names = True if df.keys().dtype == "object" else False
        df.to_csv(sys.stdout, index=False, header=keep_col_names)
