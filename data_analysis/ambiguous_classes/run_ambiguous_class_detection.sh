#!/bin/bash

DATASET_DIR=/extra/wayne1/preserve/UCI-ML-conversion
TEST_RESULTS_DIR=/extra/wayne1/preserve/UCI-ML-conversion/test-results-04-21-2025

pvalues_file=$1
features_file="${pvalues_file#${TEST_RESULTS_DIR}/}"
features_file="${features_file%%/*}"
replace="/"
features_file="${features_file/_/${replace}}"
features_file="${DATASET_DIR}/${features_file}.csv"
result_folder=${pvalues_file%/*}
python3 ambiguous_class_detection.py ${pvalues_file} ${features_file} 0.05 > "${result_folder}/ambiguous_classes.txt"
