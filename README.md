# Pidentify

Pidentify is an API which aims at classifying with P-value. This project will make classifier predict whether a test point belongs to any trained catagory.

If not, classifier will say "none of above".

## Nonlinear square fitting

Apply nonlinear square fitting to find a best curve (sigmoid function) for ECDF points in each class.

Use "alglib" library: https://www.alglib.net/interpolation/leastsquares.php#header4

Command for compiling nonlinear.ccp: g++ -I alglib/src -o nonlinear nonlinear.cpp alglib.a

function_cx_1_func contains origianl sigmoid functions. function_cx_1_grad contains derivitives of sigmoid functions in terms of array c (parameters/values to get a best fit)

rep.terminationtype: a status code returned

rep.wrmserror: residual with weights
