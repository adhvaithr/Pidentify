library("utils")
library("pvclass")

# Read dataset from CSV file into dataframe
args <- commandArgs(trailingOnly = TRUE)
data <- read.csv(args[1])

# Split the dataset into X and Y data
first_nonnum_col <- match("nonNum", substr(colnames(data), 1, 6))
if (!is.na(first_nonnum_col)) {
	X_data = data[, 2:(first_nonnum_col - 1)]
} else {
	X_data = data[, -1]
}
Y_data = data[, 1]

# Compute cross validated pvalues using weighted nearest neighbor test statistic
# Default values from the pvclass library are used for tau
pvalues <- cvpvs(X_data, Y_data, method = "wnn", wtype = "exponential")

# Display results against an alpha, where the default value is 0.05
analyze.pvs(pvalues, Y_data, alpha = 0.05, pvplot = FALSE, roc = FALSE)