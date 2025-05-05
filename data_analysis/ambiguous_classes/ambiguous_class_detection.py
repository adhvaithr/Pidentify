import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict


def find_ambiguous_classes(pvalues_file: str, features_file: str, pvalue_threshold: float) -> None:
    """
    Given the p values and features for a dataset, output a summary of the percentage of datapoints with an ambiguous
    classification per class, and what the ambiguous classifications are, along with their percentage of the
    total number of ambiguous classifications per class.
    """
    if not Path(pvalues_file).exists() or not Path(features_file).exists():
        return

    pvalues = pd.read_csv(pvalues_file)
    features = pd.read_csv(features_file, usecols=["className"])
    
    # Map class to the number of instances of that class that have ambiguous classification
    class_ambiguities = defaultdict(int)
    # Map class to the ambiguous classes for its instances and the count
    between_class_ambiguities = defaultdict(lambda: defaultdict(int))
    # Map class to total number of instances
    class_count = defaultdict(int)

    class_names = pvalues.columns[1:]
    for i in range(len(pvalues)):
        ambiguous_classes = []
        for class_name in class_names:
            if pvalues.loc[i, class_name] >= pvalue_threshold:
                ambiguous_classes.append(class_name)

        actual_class = str(features.loc[int(pvalues.loc[i, "lineNumber"]) - 2, "className"])
        class_count[actual_class] += 1
        
        if len(ambiguous_classes) > 1:
            class_ambiguities[actual_class] += 1
            between_class_ambiguities[actual_class]["/".join(ambiguous_classes)] += 1

    print(f"Ambiguous Classifications Using P Value Threshold {pvalue_threshold}\n")
    for class_name in class_names:
        percent_ambiguous = (class_ambiguities[class_name] / class_count[class_name]) * 100
        print(f"Class name: {class_name}")
        print(f"Total instances: {class_count[class_name]}")
        print(f"Percentage of ambiguous classifications: {round(percent_ambiguous, 2)}%")
        print(f"Ambiguities (# ambiguous classification/# total ambiguous classifications for {class_name}): ")
        for ambiguous_classes, count in between_class_ambiguities[class_name].items():
            print(f"\t{ambiguous_classes}: {round((count / class_ambiguities[class_name]) * 100, 2)}%")
        print()
    

if __name__ == "__main__":
    pvalues_file, features_file, pvalue_threshold = sys.argv[1:]
    find_ambiguous_classes(pvalues_file, features_file, float(pvalue_threshold))
