# UCI ML Converter

A command-line utility for converting UCI-style datasets into a cleaner tabular CSV format for downstream use in Pidentify. The converter can read multiple input formats, normalize numeric columns, preserve or infer non-numeric metadata columns, create or merge class labels, merge auxiliary files, and emit the processed dataset to standard output. :contentReference[oaicite:0]{index=0}

## Files

- `UCI_ml_converter.py` — entry point; parses CLI arguments, creates a logger, and runs dataset processing.
- `parser.py` — parses required arguments and optional flags such as row dropping, class handling, delayed output, column merging, and non-numeric inference. 
- `datafile.py` — reads and transforms each source file into a normalized pandas DataFrame.
- `dataset.py` — processes one or more `DataFile` objects and writes the result to standard output. 
- `logger.py` — creates the log file used during conversion.
- `preprocess.py` — helper for preprocessing `.Z` compressed files before conversion.

## What the converter does

The converter can:

- read `.csv`, `.data`, `.xlsx`, `.arff`, generic text tables, and `.gz`-compressed files; `.arff` files are loaded with `scipy.io.arff`, and `.gz` files are decompressed before reading. 
- drop columns, skip rows at the top, and drop rows from the footer before further processing. 
- preserve or create headers depending on `include_header`. If a header is needed, numerical columns are named `col0`, `col1`, etc., ignored non-numeric columns are named `nonNumCol...`, and the class column is named `className`. 
- remove non-numeric values from columns that are expected to be numeric, then drop rows containing invalid numeric entries. 
- create the class label from an existing class column, from the input filename, from a custom class name, or by merging several one-hot class columns into a single `className` column. 
- concatenate extra files column-wise with `--merge`, or inner join on shared column names with `--merge_on`. 
- optionally combine all rows in a file into a single row with `--combine_rows`. 
- optionally delay writing until all files are processed, then concatenate them into one output dataset. 

## Output behavior

The processed dataset is written to stdout by default. When `--delay_write` is not used, each processed file is written as soon as it is converted. When `--delay_write` is enabled, all processed DataFrames are concatenated and written once at the end. Conversion status is logged to `logs/dataset_converter_log.txt` relative to the converter directory. :contentReference[oaicite:15]{index=15}

## Command-line usage

```bash
python UCI_ml_converter.py [-H] include_header class_col ignore_cols \
  [-d rows] [-db rows] [-cls custom_class] [-clsd drop_chars] \
  [--delay_write] [--drop_col cols] [--merge_cls cols] \
  [--combine_rows] [--infer_nn] \
  input_file(s)/folder(s) [--merge file(s)] [--merge_on cols file(s)] [-rm file(s)]
