#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "interpolation.h"

using namespace alglib;
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr) 
{
    // this callback calculates f(c,x)=exp(-c0*sqr(x0))
    // where x is a position on X-axis and c is adjustable parameter
    func = exp(-c[0]*pow(x[0],2));
}
void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr) 
{
    // this callback calculates f(c,x)=exp(-c0*sqr(x0)) and gradient G={df/dc[i]}
    // where x is a position on X-axis and c is adjustable parameter.
    // IMPORTANT: gradient is calculated with respect to C, not to X
    func = exp(-c[0]*pow(x[0],2));
    grad[0] = -pow(x[0],2)*func;
}
int main(int argc, char **argv)
{
    try
    {

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
        real_2d_array x = "[[-1],[-0.8],[-0.6],[-0.4],[-0.2],[0],[0.2],[0.4],[0.6],[0.8],[1.0]]";
        real_1d_array y = "[0.223130, 0.382893, 0.582748, 0.786628, 0.941765, 1.000000, 0.941765, 0.786628, 0.582748, 0.382893, 0.223130]";
        real_1d_array c = "[0.3]";
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
    }
    catch(alglib::ap_error alglib_exception)
    {
        printf("ALGLIB exception with message '%s'\n", alglib_exception.msg.c_str());
        return 1;
    }
    return 0;
}