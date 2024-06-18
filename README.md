# Pidentify

Pidentify is an API designed to compute the variance between input data and training datasets. 

It involves introducing a "p-value" component, which will contribute to determining the p-value for each class, enabling the evaluation of input data deviation from the trained datasets. Our objective is to equip researchers with a robust tool that can improve classifier accuracy and streamline validation procedures.

Only .csv files and numerical data are supported now. Please also make sure the format of imported datasets is correct. The name of class needs to be at the end of each row. (You can use iris.data as a reference).

## How to use

Command for compiling "main.cpp", "process.cpp", and "fit.cpp": g++ --std=c++11 -I alglib/src -o exe main.cpp process.cpp fit.cpp alglib.a

Execute the compiled file: ./exe

After executing the compiled file, it will print out all best fit values along with residuals for each sigmoid functions and the best fit value among all functions.

## Store data in csv file in the data structure (main.cpp)

"main.cpp" will turn csv file into "std::vector<ClassMember> dataset" for future use.

## Data processing (process.cpp)

"process.cpp" will normalize all features, and compute the nearest neighbor distances by using KNN (k = 1 here).

And then, it will sort all distances in an ascending order and eliminate duplicated results.

## Nonlinear square fitting (fit.cpp)

"fit.cpp" is based on "alglib" library: https://www.alglib.net/interpolation/leastsquares.php#header4

Apply nonlinear square fitting to find a best value for sigmoid functions in each class.

We assume there are 2 parameters (c & a in c(x-a) in function) in each sigmoid funciton need to be tailored for ECDF points. In "fit.cpp", real_1d_array c holds the initial values for c & a. (c[0] stands for c, c[1] stands for a).

5 sigmoid functions are supported right now. They are: Logistic function, hyperbolic tangent function, arctangent function, gudermannian function, and simple algebraic function.

"function name_f" (e.g. logistic_f) stands for the original function. "function name_fd" (e.g. logistic_fd) stands for the derivative of the corresponding original function in terms of a & c.

rep.terminationtype: a status code returned

rep.wrmserror: residual with weights

You can uncomment the codes for fitting procedure to show the whole fitting process for each sigmoid function.
