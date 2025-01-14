import unlzw3
import sys
from pathlib import Path
import csv
import os
from dataclasses import dataclass


@dataclass
class Preprocessor:
    """
    Represent preprocessor, which changes file organization to be usable by the converter.
    """
    filepath: str
    file_extension: str
    destination: str

    def preprocess(self) -> None:
        """
        Modify file to be usable by converter, and write result to the destination.
        """
        if self.file_extension == ".Z":
            self._process_z_file()


    def _process_z_file(self) -> None:
        """
        Open and read .Z files.
        """
        uncompressed_data = unlzw3.unlzw(Path(filepath))
        data = uncompressed_data.decode('utf-8').splitlines()
        with open(destination, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for line in data:
                writer.writerow(line.split(","))


if __name__ == "__main__":
    filepath = sys.argv[1]
    file_extension = os.path.splitext(filepath)[-1]
    destination = sys.argv[2]
    Preprocessor(filepath, file_extension, destination).preprocess()
