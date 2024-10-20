from dataclasses import dataclass
import pandas as pd
import os
import sys

# local file paths to test with
# test/ datasets/wine\ quality/winequality-red.csv
# "test datasets"/"wine quality"/winequality-white.csv
# local-winequality-red.csv


@dataclass
class DataFile:
    """Object for each file containing data.  Note that -1 for class_col_index indicates to get the class name
    from the filename."""
    file_path: str
    file_type: str
    class_col_index: int
    is_training: bool
    ignore_cols: list[int]
    include_header: bool
    delimiter: str = None


def process_arguments():
    """Processes command line arguments into list of Datafile objects"""
    args = sys.argv
    include_header = bool(int(args[1]))
    class_col = int(args[2]) if args[2] else -1
    ignore_cols = list(map(int, args[3].split(','))) if args[3] else []
    is_training = bool(int(args[4]))
    input_files = args[5:]

    extensions = list(map(get_file_extension, input_files))

    datafile_list = [DataFile(file_path=input_files[i], file_type=extensions[i],class_col_index=class_col, ignore_cols=ignore_cols, is_training=is_training, include_header=include_header) for i in range(len(input_files))]
    return datafile_list

def load_file_objs():
    """Prompts user for file paths where data is stored, and returns an array with DataFile objects"""
    file_objs = []
    print(process_arguments())
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
    """Returns file extension of said file, taking the file path as an argument"""
    try:
        return os.path.splitext(file_path)[-1]
    except ValueError:
        print("Error: Function argument must be a file path. Retry again.")

def read_file(file_obj: DataFile):
    """Reads file depending on the file_type of the object and returns DataFrame with data once done reading file"""
    try:
        if file_obj.file_type in [".csv", ".data"]:
            return process_header(file_obj, pd.read_csv(file_obj.file_path, header=None, sep=None, engine='python'))
        elif file_obj.file_type in [".xlsx"]:
            #TODO: Make sure headers work properly with Excel files.
            return process_header(file_obj, pd.read_excel(file_obj.file_path))
        else:
            # Confirmed to work with .txt, .arff, and .dat file extensions
            return process_header(file_obj, pd.read_table(file_obj.file_path, header=None, sep=None, engine='python'))
    except pd.errors.ParserError as e:
        return e

def process_header(file_obj: DataFile, df: pd.DataFrame) -> pd.DataFrame:
    """Add a header if one is needed, or drop the header if it is present and not needed."""
    has_header = False
    non_num_vals = 0
    for i, entry in enumerate(df.iloc[0]):
        try:
            float(entry)
        except ValueError:
            non_num_vals += 1
    if non_num_vals == len(df.iloc[0]):
        has_header = True
    if has_header and not file_obj.include_header:
        df.drop(0, inplace=True)
    elif not has_header and file_obj.include_header:
        new_header = create_header(file_obj, df)
        header_map = dict(zip(range(len(df.iloc[0])), new_header))
        df.rename(columns=header_map,  inplace=True)
    
    return df

def create_header(file_obj: DataFile, df: pd.DataFrame) -> list[str]:
    """Create a header for a file that does not have one.  Numerical columns are labelled col0, col1, etc.;
    non-numerical columns are labelled nonNumColi, nonNumCol(i+1), etc. such that i is the subsequent index
    after the last numerical column; and the class column is labelled 'className'."""
    header = []
    num_cols_idx = 0
    non_num_cols_idx = len(df.iloc[0]) - (len(file_obj.ignore_cols))
    non_num_cols_idx = non_num_cols_idx - 1 if file_obj.class_col_index != -1 else non_num_cols_idx
    for col in range(len(df.iloc[0])):
        if col == file_obj.class_col_index:
            header.append("className")
        elif col in file_obj.ignore_cols:
            header.append(f"nonNumCol{non_num_cols_idx}")
            non_num_cols_idx += 1
        else:
            header.append(f"col{num_cols_idx}")
            num_cols_idx += 1
    
    return header

# TODO: Put this code fragment somewhere useful
# It gets the name of the class from the filename for a DataFile object called 'file_obj'
# os.path.basename(file_obj.file_path).split(".")[0]

def process_files(file_objs: list[DataFile]):
    """Loads and concatenates all csv files in file_paths and returns dataframe"""
    # TODO: clean up datasets with incorrectly formatted points
    loaded_dfs = []
    for file_obj in file_objs: # looping through file_objs list created in load_file_objs
        df = read_file(file_obj)
        df.dropna(inplace = True)
        df.drop_duplicates(inplace = True)
        df = move_class_to_last_column(df, file_obj.class_col_index) 
        loaded_dfs.append(df) # reads file first, then cleans and sorts columns in dataframe, then adds dataframe to list
    return pd.concat(loaded_dfs, ignore_index=True) # combines dataframes in list into one
    # TODO: Verify that dropping duplicates will remove redundant column names when there are multiples files
    # passed at the same time and decide whether to use the commented out code below
    '''
    final_df = pd.concat(loaded_dfs, ignore_index=True)
    final_df.drop_duplicates(inplace = True) 
    return final_df 
    '''

def move_class_to_last_column(df, class_col_ind: int):
    """Swaps class column at index class_col_ind with the last column in the df Dataframe"""
    columns = list(df.columns)
    columns[class_col_ind], columns[-1] = columns[-1], columns[class_col_ind] # moves the class column to the last column in the dataframe
    return df[columns]

def valid_file_name(desired_name: str):
    """Check that the characters within desired name do not include special characters with ascii"""
    valid_characters = [32, 95] + list(range(48, 58)) + list(range(65, 91)) + list(range(97, 123))
    for char in desired_name:
        ascii_char = ord(char)
        if ascii_char not in valid_characters:
            return False
    return True

def get_output_file_name():
    """Get desired name from user for output CSV file"""
    name = input("Enter desired name for output file: ").strip()
    while not (name and valid_file_name(name)):
        if not name:
            print("File name can't be empty or contain only whitespace.")
        else:
            print("File name can't contain special characters besides space and underscore.")
        name = input("Enter desired name for output file: ").strip()
    if (" " in name):
        name = name.replace(" ", "_")
    return name

def create_output_file(df, file_name):
    """Creates a csv file called filename containing combined dataframe"""
    df.to_csv(file_name, index=False)

def main():
    # file_objs = load_file_objs()
    # output_file_name = get_output_file_name()
    
    file_objs = process_arguments()
    df = process_files(file_objs)
    keep_col_names = True if df.keys().dtype == "object" else False
    df.to_csv(sys.stdout, index=False, header=keep_col_names)
    # create_output_file(df, output_file_name)


if __name__ == "__main__":
    main()
