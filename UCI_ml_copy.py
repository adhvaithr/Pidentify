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


def print_help():
    print("""python UCI_ml_copy.py [-H] include_header class_col ignore_cols training input_file(s)
    \t-H\t\tOptional. Prints the usage statement. All other arguments are ignored.
    \tinclude_header\tInteger. If 1, add header if not already in dataset. If 0, drop header if dataset has one.
    \tclass_col\tInteger. The index of the classification column.
    \tignore_col(s)\tInteger(s). A list of indexes seperated by comma. Ex. '0,1,2,3,4,5,6'.
    \ttraining\tInteger. If 1, following files are training datasets. If 0, following files are testing datasets.
    \tinput_file(s)\tFile paths. The remaining arguments are paths to files. Invalid file paths are ignored.""")


def check_arguments(argument_list: list):
    if (len(argument_list) == 1):
        print("No arguments provided.")
        print_help()
        sys.exit(1)
    else:
        if (argument_list[1] == "-H"):
            print_help()
            sys.exit(0)
        elif (len(argument_list) < 6):
            print("Not enough arguments.")
            print_help()
            sys.exit(1)


def process_arguments() -> DataFile:
    """
    Processes command line arguments into list of Datafile objects
    
    :returns: DataFile object with all arguments to be used in later commands
    :rtype: DataFile
    """

    args = sys.argv
    check_arguments(args)
    include_header = bool(int(args[1]))
    class_col = int(args[2]) if args[2] else -1
    ignore_cols = list(map(int, args[3].split(','))) if args[3] else []
    is_training = bool(int(args[4]))
    input_files = args[5:]

    extensions = list(map(get_file_extension, input_files))

    datafile_list = [DataFile(file_path=input_files[i], file_type=extensions[i], class_col_index=class_col,
                              ignore_cols=ignore_cols, is_training=is_training, include_header=include_header)
                     for i in range(len(input_files))]
    return datafile_list


def get_file_extension(file_path: str) -> str:
    """
    Returns file extension of said file, taking the file path as an argument

    :param str file_path: path of a file
    :returns: file extension
    :rtype: str
    """
    try:
        return os.path.splitext(file_path)[-1]
    except ValueError:
        print("Error: Function argument must be a file path. Retry again.")


def read_file(file_obj: DataFile) -> pd.DataFrame:
    """
    Reads file depending on the file_type of the object and returns DataFrame with data once done reading file.
    
    :param DataFile file_obj: file object of DataFile class
    :returns: Dataframe read by Pandas depending on extension of file.
    :rtype: pd.DataFrame
    """
    try:
        if file_obj.file_type in [".csv", ".data"]:
            return process_header(file_obj, pd.read_csv(file_obj.file_path, header=None, sep=None, engine='python'))
        elif file_obj.file_type in [".xlsx"]:
            # TODO: Make sure headers work properly with Excel files.
            return process_header(file_obj, pd.read_excel(file_obj.file_path))
        else:
            # Confirmed to work with .txt, .arff, and .dat file extensions
            return process_header(file_obj, pd.read_table(file_obj.file_path, header=None, sep=None, engine='python'))
    except pd.errors.ParserError as e:
        return e


def process_header(file_obj: DataFile, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a header if one is needed, or drop the header if it is present and not needed.

    :param DataFile file_obj: file_obj (checks if header is present)
    :param pd.DataFrame df: dataset that might require a header
    :returns: formatted dataframe with header (if needed)
    :rtype: pd.DataFrame
    """

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
    elif has_header and file_obj.include_header:
        modified_header = []
        for i, col_name in enumerate(df.iloc[0]):
            if i in file_obj.ignore_cols:
                modified_header.append("nonNum" + col_name)
            else:
                modified_header.append(col_name)
        header_map = dict(zip(range(len(df.iloc[0])), modified_header))
        df.rename(columns=header_map,  inplace=True)
        df.drop(0, inplace=True)
    elif not has_header and file_obj.include_header:
        new_header = create_header(file_obj, df)
        header_map = dict(zip(range(len(df.iloc[0])), new_header))
        df.rename(columns=header_map, inplace=True)

    return df


def create_header(file_obj: DataFile, df: pd.DataFrame) -> list[str]:
    """
    Create a header for a file that does not have one.  Numerical columns are labelled col0, col1, etc.;
    non-numerical columns are labelled nonNumCol(i), nonNumCol(i+1), etc. such that i is the subsequent index
    after the last numerical column; and the class column is labelled 'className'.
    
    :param DataFile file_obj: file_obj (checks if header is present)
    :param pd.DataFrame df: dataset that might require a header
    :returns: header columns for my file right now
    :rtype: list[str]
    """

    header = []
    num_cols_idx = 0
    non_num_cols_idx = len(df.iloc[0]) - (len(file_obj.ignore_cols))
    non_num_cols_idx = non_num_cols_idx - \
        1 if file_obj.class_col_index != -1 else non_num_cols_idx
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


def process_files(file_objs: list[DataFile]) -> pd.DataFrame:
    """
    Loads and concatenates all csv files in file_paths and returns dataframe

    :param list file_objs: multiple data files 
    :returns: concatenated dataset
    :rtype: pd.DataFrame
    """
    # TODO: clean up datasets with incorrectly formatted points
    loaded_dfs = []
    for file_obj in file_objs:
        df = read_file(file_obj)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        if file_obj.class_col_index != -1:
            df = move_class_to_first_column(df, file_obj.class_col_index)
        else:
            df.insert(0, "className", [os.path.basename(file_obj.file_path).split(".")[0]] * len(df))
        df = move_non_num_columns_to_last(file_obj.class_col_index, df, file_obj.ignore_cols)
        # reads file first, then cleans and sorts columns in dataframe, then adds dataframe to list
        loaded_dfs.append(df)
    # combines dataframes in list into one
    return pd.concat(loaded_dfs, ignore_index=True)
    # TODO: Verify that dropping duplicates will remove redundant column names when there are multiples files
    # passed at the same time and decide whether to use the commented out code below
    '''
    final_df = pd.concat(loaded_dfs, ignore_index=True)
    final_df.drop_duplicates(inplace = True) 
    return final_df 
    '''

def move_class_to_first_column(df, class_col_ind: int):
    """"
    Swaps class column at index class_col_ind with the first column in the df Dataframe

    :param pd.DataFrame df: unformatted dataset
    :param int class_col_ind: location of the class column
    :returns: formatted dataset with class column at first column
    :rtype: pd.DataFrame
    """
    columns = list(df.columns)
    target_name = columns[class_col_ind]
    first_column = df.pop(target_name)
    df.insert(0, target_name, first_column)
    return df

def move_non_num_columns_to_last(prev_class_index, df, ignored_cols: list):
    """
    Moves all non num columns to the end of the dataframe

    :param prev_class_index: the previous index for the class name
    :param pd.DataFrame df: unformatted dataset
    :param ignored_cols: a list of the columns we want to ignore
    :returns: formatted dataset with all non columns at the end
    :rtype: pd.DataFrame
    """
    columns = list(df.columns) #target in index 
    if prev_class_index == -1:
        new_ignored_cols = [i+1 for i in ignored_cols]
    else:
        new_ignored_cols = [i+1 for i in ignored_cols if i < prev_class_index]
    ignored_name_cols = [columns[i] for i in new_ignored_cols]
    df = df[[col for col in columns if col not in ignored_name_cols] + ignored_name_cols]
    return df

def valid_file_name(desired_name: str) -> bool:
    """
    Check that the characters within desired name do not include special characters with ascii

    :param str desired_name: tentative name of file
    :returns: Does it include special characters or not?
    :rtype: bool
    """

    valid_characters = [
        32, 95] + list(range(48, 58)) + list(range(65, 91)) + list(range(97, 123))
    for char in desired_name:
        ascii_char = ord(char)
        if ascii_char not in valid_characters:
            return False
    return True


def main():
    file_objs = process_arguments()
    df = process_files(file_objs)
    keep_col_names = True if df.keys().dtype == "object" else False
    df.to_csv(sys.stdout, index=False, header=keep_col_names)


if __name__ == "__main__":
    main()

"""
def move_non_num_columns_to_last(prev_class_index, df, ignored_cols: list):
    columns = list(df.columns) #target in index 0
    new_ignored_cols = [i+1 for i in ignored_cols if i < prev_class_index]
    ignored_name_cols = [columns[i] for i in new_ignored_cols]
    df = df[[col for col in columns if col not in ignored_name_cols] + ignored_name_cols]
    return df

def move_class_to_first_column(df, class_col_ind: int):
    Swaps class column at index class_col_ind with the last column in the df Dataframe
    columns = list(df.columns)
    target_name = columns[class_col_ind]
    first_column = df.pop(target_name)
    df.insert(0, target_name, first_column)
    return df
"""