# Pidentify

Pidentify is an API designed to compute the variance between input data and training datasets. 

It involves introducing a "p-value" component, which will contribute to determining the p-value for each class, enabling the evaluation of input data deviation from the trained datasets. Our objective is to equip researchers with a robust tool that can improve classifier accuracy and streamline validation procedures.

Only .csv files and numerical data are supported in this API, and please also make sure imported datasets have unified formats (For example. ID is the class in first column).

## How it works

API will make classifier predict whether a test point belongs to any trained catagory.

If given input can be categorized into training sets, prediction works correctly. If not, make classifier stop make predictions (by saying "none of above").

## Nonlinear square fitting

Apply nonlinear square fitting to find a best curve (sigmoid function) for ECDF points in each class.

Use "alglib" library: https://www.alglib.net/interpolation/leastsquares.php#header4

Command for compiling nonlinear.ccp: g++ --std=c++11 -I alglib/src -o exe main.cpp process.cpp fit.cpp alglib.a

Execute the compiled file: ./exe

function_cx_1_func contains origianl sigmoid functions. function_cx_1_grad contains derivitives of sigmoid functions in terms of array c (parameters/values to get a best fit)

rep.terminationtype: a status code returned

rep.wrmserror: residual with weights
