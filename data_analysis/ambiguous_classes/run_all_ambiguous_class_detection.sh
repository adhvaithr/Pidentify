#!/bin/bash

run_all() {
	subfolder=$1
	for folder in /extra/wayne1/preserve/UCI-ML-conversion/test-results-04-21-2025/*; do if [ -f "${folder}/${subfolder}/summary.csv" ]; then bash run_ambiguous_class_detection.sh ${folder}/${subfolder}/pvalues.csv; fi; done
}

run_incomplete() {
	subfolder=$1
	for folder in /extra/wayne1/preserve/UCI-ML-conversion/test-results-04-21-2025/*; do if [ -f "${folder}/${subfolder}/summary.csv" ] && ! [ -f "${folder}/${subfolder}/ambiguous_classes.txt" ]; then bash run_ambiguous_class_detection.sh ${folder}/${subfolder}/pvalues.csv; fi; done
}

run_selection=$1
if [ "${run_selection}" == "--run_all" ]; then
	run_all "PCA"
	run_all "PCA/nonPCA"
elif [ "${run_selection}" == "--run_incomplete" ]; then
	run_incomplete "PCA"
	run_incomplete "PCA/nonPCA"
else
	echo "Pass --run_all to run on all datasets, or --run_incomplete to run only on datasets missing an ambiguous_classes.txt file."
fi
