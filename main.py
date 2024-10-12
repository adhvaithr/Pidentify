from dataclasses import dataclass
import pandas as pd
import os


# local file paths to test with
# test\ datasets/wine\ quality/winequality-red.csv
# "test datasets"/"wine quality"/winequality-white.csv
# local-winequality-red.csv


@dataclass
class DataFile:
    """Object for each file containing data"""
    file_path: str
    file_type: str
    class_col_index: int
    delimiter: str = None

def load_file_objs():
    """Prompts user for file paths where data is stored, and returns an list with DataFile objects"""
    file_objs = []

    while True:
        file_path = input("Enter file path to data file (press enter to exit): ").strip()

        # Break out of while loop if empty input  
        if not file_path:
            break     

        # Discard file path input if no existing file found
        if not os.path.exists(file_path):
            print("Error: File not found.")
            continue
        try:
            # Prompt user for index of class column
            class_col = int(input("Enter column index (starting at 0) of class column: ").strip())
            # TODO: check column in valid range
        except ValueError:
            print("Error: Class column must be integer. File not loaded")
            continue
        
        # Get file type
        extension = get_file_extension(file_path)

        # Create DataFile object and add to file_objs list
        file_obj = DataFile(file_path = file_path, file_type=extension, class_col_index = class_col)
        file_objs.append(file_obj)

    return file_objs

def get_file_extension(file_path: str):
<<<<<<< Updated upstream
    """Returns file extension of said file, taking the file path as an argument"""
=======
    """Returns file extension"""
    # TODO: add error handling
>>>>>>> Stashed changes
    try:
        return os.path.splitext(file_path)[-1]
    except ValueError:
        print("Error: Function argument must be a file path. Retry again.")

def read_file(file_obj: DataFile):
    """Reads file depending on the file_type of the object and returns DataFrame with data once done reading file"""
    try:
        if file_obj.file_type in [".csv", ".data", ".txt"]:
            return pd.read_csv(file_obj.file_path, sep=None, engine='python')
        elif file_obj.file_type in [".xlsx"]:
            return pd.read_excel(file_obj.file_path, sep=None, engine='python')
    except pd.errors.ParserError as e:
        return e
    

def process_files(file_objs: list[DataFile]):
    """Loads and concatenates all csv files in file_paths and returns dataframe"""
    # TODO: what if files in file_paths are diff format???
    # TODO: clean up datasets with points like N/A
    loaded_dfs = []
    for file_obj in file_objs:
        df = read_file(file_obj)
        df.dropna(inplace = True)
        df.drop_duplicates(inplace = True)
        df = move_class_to_last_column(df, file_obj.class_col_index)
        loaded_dfs.append(df)
    return pd.concat(loaded_dfs, ignore_index=True)

def move_class_to_last_column(df, class_col_ind: int):
    """Swaps class column at index class_col_ind with the last column in the df Dataframe"""
    columns = list(df.columns)
    columns[class_col_ind], columns[-1] = columns[-1], columns[class_col_ind]
    return df[columns]

def get_output_file_name():
    """Get desired name from user for output CSV file"""
    name = input("Enter desired name for output file: ").strip()
    while not name:
        name = input("Enter desired name for output file: ").strip()
    # TODO: check if valid file name. Currently only making sure that some value is given
    return name

def create_output_file(df, file_name):
    """Creates a csv file called filename containing df dataframe """
    df.to_csv(file_name)

def main():
    file_objs = load_file_objs()
    output_file_name = get_output_file_name()
    df = process_files(file_objs)
    create_output_file(df, output_file_name)


if __name__ == "__main__":
    main()