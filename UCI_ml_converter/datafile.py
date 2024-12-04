from dataclasses import dataclass
import pandas as pd
import os
import numpy as np
import gzip
import shutil
from scipy.io import arff


@dataclass
class DataFile:
    """Object for each file containing data.  Note that -1 for class_col_index indicates to get the class name
    from the filename."""
    file_path: str
    file_type: str
    class_col_index: int
    ignore_cols: list[int]
    drop_cols: list[int]
    ignore_rows: int
    drop_footer: int
    include_header: bool
    class_drop_chars: bool
    custom_class: list[str]
    merge_class: list[int]
    infer_non_num: bool = False
    delimiter: str = None


    def process_file(self, *files, merge: bool = False) -> pd.DataFrame:
        """
        Load a single file and return a dataframe with non-numerical values removed from numerical columns
        and the columns reordered.
        """
        df = self._read_file()
        if merge:
            df = DataFile._merge_files(df, *files)
        if self.drop_cols:
            self._remove_cols(df)
        df = self._process_header(df)
        # Numerical columns with the class column excluded, since it can be numerical or non-numerical
        num_cols = sorted(list(set(range(len(df.columns))) - set(self.ignore_cols) - \
                          set([self.class_col_index] if self.class_col_index != -1 else [])))
        DataFile._drop_non_num(num_cols, df)
        df.dropna(subset = [df.columns[col_idx] for col_idx in num_cols], inplace=True)
        df.drop_duplicates(inplace = True)
        df = self._reorder_columns(num_cols, df)
        return df


    def _read_file(self) -> pd.DataFrame:
        """
        Reads file depending on the file_type of the object and returns DataFrame with data once done reading file.
    
        :param DataFile self: file object of DataFile class
        :returns: Dataframe read by Pandas depending on extension of file.
        :rtype: pd.DataFrame
        """
        if self.file_path.endswith(".gz"):
            uncompressed_file_path = self.file_path.replace(".gz", "")
            with gzip.open(self.file_path, "rb") as file_in:
                with open(uncompressed_file_path, "wb") as file_out:
                    shutil.copyfileobj(file_in, file_out)
            self.file_path = uncompressed_file_path
            self.file_type = self.file_type[:-3]
        if self.file_type in [".csv", ".data"]:
            return pd.read_csv(self.file_path, header=None, skiprows=self.ignore_rows, skipfooter=self.drop_footer, sep=None, engine='python')
        elif self.file_type in [".xlsx"]:
            return pd.read_excel(self.file_path)
        elif self.file_type in [".arff"]:
            data, info = arff.loadarff(self.file_path)
            df =  pd.DataFrame(data)
            df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            return df
        else:
            return pd.read_table(self.file_path, header=None, skiprows=self.ignore_rows, skipfooter=self.drop_footer, sep=None, engine='python')


    @staticmethod
    def _merge_files(df: pd.DataFrame, *files) -> pd.DataFrame:
        """
        Concatenate columns gathered from files to the end of a dataframe, and return the
        resulting dataframe.
        """
        merge_dfs = []
        for file in files:
            with open(file.file_path) as file:
                res = [line.strip() for line in file.readlines()]    
            merge_dfs.append(pd.DataFrame(res, columns=[len(df.columns)]))
        return pd.concat([df, *merge_dfs], axis=1)
    

    def _remove_cols(self, df: pd.DataFrame) -> None:
        """
        Drop columns from the file, and reindex the class column, columns being merged, and columns being
        ignored.
        """
        df.drop(columns=[df.columns[i] for i in self.drop_cols], inplace=True)
        if self.class_col_index != -1:
            self.class_col_index = DataFile._reindex_col(self.class_col_index, self.drop_cols)
        for i, merge_col in enumerate(self.merge_class):
            self.merge_class[i] = DataFile._reindex_col(merge_col, self.drop_cols)
        for i, ignore_col in enumerate(self.ignore_cols):
            self.ignore_cols[i] = DataFile._reindex_col(ignore_col, self.drop_cols)


    def _process_header(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a header if one is needed, or drop the header if it is present and not needed.

        :param DataFile self: self (checks if header is present)
        :param pd.DataFrame df: dataset that might require a header
        :returns: formatted dataframe with header (if needed)
        :rtype: pd.DataFrame
        """
        if self.infer_non_num:
            for i in range(len(df.iloc[0])):
                if df.dtypes.iloc[i] == "object":
                    self.ignore_cols.append(i)
        
        has_header = False
        non_num_vals = 0
        for i, entry in enumerate(df.iloc[0]):
            try:
                float(entry)
            except ValueError:
                non_num_vals += 1
        if non_num_vals == len(df.iloc[0]):
            has_header = True
        if has_header and not self.include_header:
            df.drop(0, inplace=True)
        elif has_header and self.include_header:
            modified_header = []
            for i, col_name in enumerate(df.iloc[0]):
                if i in self.ignore_cols:
                    modified_header.append("nonNum" + col_name)
                else:
                    modified_header.append(col_name)
            header_map = dict(zip(df.columns, modified_header))
            df.rename(columns=header_map, inplace=True)
            df.drop(0, inplace=True)
        elif not has_header and self.include_header:
            new_header = self._create_header(df)
            header_map = dict(zip(df.columns, new_header))
            df.rename(columns=header_map, inplace=True)

        return df


    def _create_header(self, df: pd.DataFrame) -> list[str]:
        """
        Create a header for a file that does not have one.  Numerical columns are labelled col0, col1, etc.;
        non-numerical columns are labelled nonNumCol(i), nonNumCol(i+1), etc. such that i is the subsequent index
        after the last numerical column; and the class column is labelled 'className'.
    
        :param DataFile self: self (checks if header is present)
        :param pd.DataFrame df: dataset that might require a header
        :returns: header columns for my file right now
        :rtype: list[str]
        """

        header = []
        num_cols_idx = 0
        non_num_cols_idx = len(df.iloc[0]) - (len(self.ignore_cols))
        non_num_cols_idx = non_num_cols_idx - \
            1 if self.class_col_index != -1 else non_num_cols_idx
        for col in range(len(df.iloc[0])):
            if col == self.class_col_index:
                header.append("className")
            elif col in self.ignore_cols:
                header.append(f"nonNumCol{non_num_cols_idx}")
                non_num_cols_idx += 1
            else:
                header.append(f"col{num_cols_idx}")
                num_cols_idx += 1

        return header


    def _create_cls_name(self) -> str:
        """
        Create the class name by either returning the filename, the filename with characters removed,
        or a custom class name.
        """
        filename = os.path.basename(self.file_path).split(".")[0]
        if self.class_drop_chars and self.custom_class:
            class_name = filename
            for drop_chars in self.custom_class:
                if drop_chars.startswith("*"):
                    class_name = class_name.split(drop_chars[1:])[-1]
                elif drop_chars.endswith("*"):
                    class_name = class_name.split(drop_chars[:-1])[0]
                else:
                    class_name = class_name.replace(drop_chars, "")
        elif self.custom_class:
            class_name = self.custom_class[0]
        else:
            class_name = filename
    
        return class_name


    @staticmethod
    def _convert_non_num_to_na(entry):
        """Return the entry if the entry is numerical and null otherwise."""
        try:
            float(entry)
        except ValueError:
            return np.NaN
        return entry


    @staticmethod
    def _drop_non_num(num_cols: list[int], df: pd.DataFrame) -> None:
        """Change all non-numerical entries in columns that are supposed to be numerical into null entries."""
        for num_col in num_cols:
            if df.dtypes.iloc[num_col] != "object":
                continue
            df.iloc[:, num_col] = df.iloc[:, num_col].apply(DataFile._convert_non_num_to_na)
    

    def _reorder_columns(self, num_cols: list[int], df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns of dataframe to be the class columns followed by the numerical columns, and all
        non-numerical columns are placed at the very end.  A class column is created if it doesn't exist.
        """
        if self.class_col_index == -1:
            class_name = self._create_cls_name()
            class_col_header = "className" if self.include_header else len(df.columns)
            if self.merge_class:
                class_col, num_cols = self._merge_class_columns(num_cols, df)
            else:
                class_col = pd.DataFrame([class_name] * len(df), index=df.index, columns=[class_col_header])
        df = df[([df.columns[self.class_col_index]] if self.class_col_index != -1 else []) + [df.columns[i] for i in num_cols] + [df.columns[i] for i in self.ignore_cols]]
        if self.class_col_index == -1:
            df = pd.concat([class_col, df], axis=1)
        return df


    def _merge_class_columns(self, num_cols: list[int], df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
        """
        Merge multiple class columns together into one class column.  It is currently assumed that the correct
        name of the class can be found in the name of the merged columns and that headers are being included.
        The ignored columns and numerical columns are reindexed to reflect the columns dropped during the merge.
        """
        def combine_columns(*col_entries):
            col_entries = list(map(int, col_entries))
            return merge_cols[col_entries.index(1)]
        merge_cols = [df.columns[i] for i in self.merge_class]
        class_col = pd.DataFrame(list(map(combine_columns, *[df[col_name] for col_name in merge_cols])),
                                 index=df.index, columns=["className"])
        df.drop(columns=merge_cols, inplace=True)
        for i, ignore_col in enumerate(self.ignore_cols):
            self.ignore_cols[i] = DataFile._reindex_col(ignore_col, self.merge_class)
        num_cols = list(set(num_cols) - set(self.merge_class))
        for i, col in enumerate(num_cols):
            if col > min(self.merge_class):
                num_cols[i] = DataFile._reindex_col(col, self.merge_class)
        return class_col, num_cols
    

    @staticmethod
    def _reindex_col(col: int, compare_list: list[int]) -> int:
        """
        Helper function to reindex columns where column indices need to have a certain offset
        subtracted from them when columns of smaller indices are deleted.
        """
        decrement = 0
        for compare_col in compare_list:
            if col > compare_col:
                decrement += 1
        return (col - decrement)
