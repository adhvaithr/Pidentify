#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "interpolation.h"

using namespace alglib;
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1.0 / (1.0 + exp(-c[0] * x[0] - c[1]));
}
void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr) {
    // (1+ e^x/(e^x+1)^2)/2
    grad[0] = (1 + exp(c[0] * x[0] + c[1]) / ((exp(c[0] * x[0] + c[1]) + 1.0) * (exp(c[0] * x[0] + c[1]) + 1.0)))/2;
}

int main(int argc, char **argv)
{
    try
    {
        // kx + b
        // ((kx + b) +1) / 2
        //   (k (cx + a) + b + 1) / 2
        //
        // In this example we demonstrate exponential fitting by
        //
        //     f(x) = exp(-c*x^2)
        //
        // using function value and gradient (with respect to c).
        //
        // IMPORTANT: the LSFIT optimizer supports parallel model  evaluation  and
        //            parallel numerical differentiation ('callback parallelism').
        //            This feature, which is present in commercial ALGLIB editions
        //            greatly  accelerates  fits  with   large   datasets   and/or
        //            expensive target functions.
        //
        //            Callback parallelism is usually  beneficial  when  a  single
        //            pass over the entire  dataset  requires  more  than  several
        //            milliseconds. This particular example,  of  course,  is  not
        //            suited for callback parallelism.
        //
        //            See ALGLIB Reference Manual, 'Working with commercial version'
        //            section,  and  comments  on  lsfitfit()  function  for  more
        //            information.
        //
        real_2d_array x = "[[0],[0.11],[0.21],[0.34],[0.42],[0.53],[0.55],[0.56],[0.58],[0.59],[0.90]]";
        real_1d_array y = "[0, 0.05, 0.1, 0.15, 0.20, 0.23, 0.44, 0.45, 0.5, 0.53, 0.75]";
        real_1d_array c = "[0.1, 0.2]";
        double epsx = 0.000001;
        ae_int_t maxits = 0;
        lsfitstate state;
        lsfitreport rep;
        
        //
        // Fitting with weights
        // (you can change weights and see how it changes result)
        //
        real_1d_array w = "[1,1,1,1,1,1,1,1,1,1,1]";
        lsfitcreatewfg(x, y, w, c, state);
        lsfitsetcond(state, epsx, maxits);
        alglib::lsfitfit(state, function_cx_1_func, function_cx_1_grad);
        lsfitresults(state, c, rep);
        printf("%d\n", int(rep.terminationtype)); // EXPECTED: 2
        printf("%s\n", c.tostring(1).c_str()); // EXPECTED: [1.5]
        printf("%g\n", rep.wrmserror);
    }
    catch(alglib::ap_error alglib_exception)
    {
        printf("ALGLIB exception with message '%s'\n", alglib_exception.msg.c_str());
        return 1;
    }
    return 0;
}