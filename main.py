from dataclasses import dataclass
import pandas as pd
import os


# local file paths to test with
# ./test_datasets/iris/iris.data
# ./test_datasets/wine+quality/winequality-red.csv'
# ./test_datasets/wine+quality/winequality-white.csv'

@dataclass
class DataFile:
    "Object for each file containing data"
    file_path: str
    file_type: str
    class_col_index: int
    # delimiter: str   TODO: delimiter field??

def load_file_objs():
    """Prompts user for file paths where data is stored"""
    file_objs = []
    while True:
        file_path = input("Enter file path to data file (press enter to exit): ").strip()     
         
        if not file_path:
            break     

        if not os.path.exists(file_path):
            print("Error: File not found.")
            continue
        try:
            class_col = int(input("Enter column index (starting at 0) of class column: ").strip())
            # TODO: check column in valid range
        except ValueError:
            print("Error: Class column must be integer. File not loaded")
            continue

        extension = get_file_extension(file_path)
        file_obj = DataFile(file_path = file_path, file_type=extension, class_col_index = class_col)
        file_objs.append(file_obj)

    return file_objs

def get_file_extension(file_path: str):
    """Returns file extension"""
    # TODO: add error handling
    extension = os.path.splitext(file_path)[-1]
    return extension

def read_file(file_obj: DataFile):
    """Reads file and returns Dataframe"""
    # file_type = get_file_extension(file_path)
    try:
        if file_obj.file_type in [".csv", ".data"]:
            return pd.read_csv(file_obj.file_path, sep=None, engine='python')
        elif file_obj.file_type in [".xlsx"]:
            return pd.read_excel(file_obj.file_path, sep=None, engine='python')
    except pd.errors.ParserError as e:
        # TODO: decide on error functionality when reading files. Currently returning empty dataframe, ignoring data in file that caused error
        return pd.DataFrame()
    

def process_files(file_objs: list[DataFile]):
    """Loads and concatenates all csv files in file_paths and returns dataframe"""
    # TODO: what if files in file_paths are diff format???
    loaded_dfs = []
    for file_obj in file_objs:
        df = read_file(file_obj)
        df = move_class_to_last_column(df, file_obj.class_col_index)
        loaded_dfs.append(df)
    return pd.concat(loaded_dfs, ignore_index=True)

def move_class_to_last_column(df, class_col_ind: int):
    """Swaps class column at column class_col_ind with the last column in the df Dataframe"""
    columns = list(df.columns)
    columns[class_col_ind], columns[-1] = columns[-1], columns[class_col_ind]
    return df[columns]

def get_output_file_name():
    """Get desired name for output csv file"""
    name = input("Enter desired name for output file: ")
    return name

def main():
    # print(get_file_extension("./test_datasets/wine+quality/winequality-red.csv"))
    file_objs = load_file_objs()
    output_file_name = get_output_file_name()
    df = process_files(file_objs)
    print(df)

if __name__ == "__main__":
    main()