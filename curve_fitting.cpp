#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "interpolation.h"

using namespace alglib;

// helper function secant
double sech(double x) {
    return 1.0 / std::cosh(x);
}

// logistic function
double logistic(double k, double alpha, double x){
     return 1.0 / (1.0 + exp(-k*(x-alpha)));
}

void logistic_f(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - logistic(c[0],c[1],x[0]);
}
void logistic_fd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr) {
    func = 1 - logistic(c[0],c[1],x[0]);
    grad[0] = - (((x[0]-c[1]) * exp(c[0] * (c[1] - x[0]))) / (exp(c[0] * (c[1] - x[0])) + 1) * (exp(c[0] * (c[1] - x[0])) + 1));
    grad[1] = c[0] * exp(c[0] * (c[1] - x[0])) / (exp(c[0] * (c[1] - x[0])) + 1) * (exp(c[0] * (c[1] - x[0])));
}

// hyperbolic tangent
double hyperbolic_tangent(double k, double alpha, double x)
{
    return ((exp(k * (x - alpha)) - exp(-k * (x - alpha)))/(exp(k * (x - alpha)) + exp(-k * (x - alpha))) + 1) / 2;
}

void hyperbolic_f(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - hyperbolic_tangent(c[0], c[1], x[0]);
}

void hyperbolic_fd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    func = 1 - hyperbolic_tangent(c[0], c[1], x[0]);
    grad[0] = - (2 * (x[0] - c[1]) * exp(2 * c[0] * (x[0] - c[1]))) / ((exp(2 * c[0] * (x[0] - c[1])) + 1) *  (exp(2 * c[0] * (x[0] - c[1])) + 1));
    grad[1] = (2 * c[0] * exp(2 * c[0] * (x[0] - c[1]))) / ((exp(2 * c[0] * (x[0] - c[1])) + 1) * (exp(2 * c[0] * (x[0] - c[1])) + 1));
}

// arctangent function
double arctangent(double k, double alpha, double x)
{
    return (atan(k * (x - alpha)) + 1) / 2;
}

void arctangent_f(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - arctangent(c[0], c[1], x[0]);
}

void arctangent_fd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    func = 1 - arctangent(c[0], c[1], x[0]);
    grad[0] = - ((x[0] - c[1]) / (2 * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1)));
    grad[1] =  - (c[0] / (2 * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1)));
}

// gudermannian function
double gudermannian(double k, double alpha, double x)
{
    return ((2 * atan(tanh(k * (x - alpha)/ 2))) + 1) / 2;
}

void gudermannian_f(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - gudermannian(c[0], c[1], x[0]);
}

void gudermannian_fd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    func = 1 - gudermannian(c[0], c[1], x[0]);
    grad[0] = -((x[0] - c[1]) * sech(1/2 * c[0] * (x[0] - c[1])) * sech(1/2 * c[0] * (x[0] - c[1])))/ (2 * ((tanh(1/2 * c[0] * (x[0] - c[1])) * tanh(1/2 * c[0] * (x[0] - c[1])) + 1)));
    grad[1] = c[0] * sech(1/2 * c[0] * (x[0] - c[1])) * sech(1/2 * c[0] * (x[0] - c[1])) / (2 * (tanh(1/2 * c[0] * (x[0] - c[1])) * tanh(1/2 * c[0] * (x[0] - c[1])) + 1));
}

// simple algebraic function
double algebraic(double k, double alpha, double x)
{
    double term = k * (x - alpha);
    return ((k * (x - alpha)) / (sqrt(1 + term * term)) + 1) / 2;
}

void algebraic_f(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - algebraic(c[0], c[1], x[0]);
}

void algebraic_fd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    func = 1 - algebraic(c[0], c[1], x[0]);
    grad[0] = - ((x[0] - c[1]) / (2 * (sqrt((c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1)))));
    grad[1] = - (c[0] / (2 * (sqrt((c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1)))));
}


int curve_fitting(std::vector<double> sorted_distances, std::vector<double> y_values)
{
    alglib::real_2d_array x;
    alglib::real_1d_array y;
    alglib::real_1d_array w;

    x.setlength(sorted_distances.size(), 1);
    y.setlength(y_values.size());

    // Copying data from vector to ALGLIB array
    for(size_t i = 0; i < sorted_distances.size(); i++) {
        x[i][0] = sorted_distances[i];  // Assuming each subvector has exactly one element
    }

    for(size_t i = 0; i < y_values.size(); i++) {
        y[i] = y_values[i];
    }

    // set weights for fitting
    w.setlength(y_values.size());
    for(size_t i = 0; i < y_values.size(); i++) {
        w[i] = sorted_distances[i]*sorted_distances[i];
    }

    try
    {
        real_1d_array c = "[0.367, 0.45]";
        double epsx = 0;
        ae_int_t maxits = 0;
        lsfitstate state;
        lsfitreport rep;

        // nonlinear square curve fitting for logistic function
        lsfitcreatewfg(x, y, w, c, state);
        lsfitsetcond(state, epsx, maxits);
        alglib::lsfitfit(state, logistic_f, logistic_fd);
        lsfitresults(state, c, rep);
        //printf("%d\n", int(rep.terminationtype));  // status code
        printf("%s\n", c.tostring(1).c_str()); // best fit values for c & a in c(x-a)
        printf("%f\n", rep.wrmserror); // residual

        // print out the fitting procedure
        /*for (int i = 0; i < y.size(); i++){
            printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - logistic(c[0], c[1], x[i][0]));
        }*/

        // nonlinear square curve fitting for hyperbolic tangent function
        lsfitcreatewfg(x, y, w, c, state);
        lsfitsetcond(state, epsx, maxits);
        alglib::lsfitfit(state, hyperbolic_f, hyperbolic_fd);
        lsfitresults(state, c, rep);
        //printf("%d\n", int(rep.terminationtype));
        printf("%s\n", c.tostring(1).c_str());
        printf("%f\n", rep.wrmserror);

        // print out the fitting procedure
        /*for (int i = 0; i < y.size(); i++){
            printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - hyperbolic_tangent(c[0], c[1], x[i][0]));
        }*/

        // nonlinear square curve fitting for arctangent function
        lsfitcreatewfg(x, y, w, c, state);
        lsfitsetcond(state, epsx, maxits);
        alglib::lsfitfit(state, arctangent_f, arctangent_fd);
        lsfitresults(state, c, rep);
        //printf("%d\n", int(rep.terminationtype));
        printf("%s\n", c.tostring(1).c_str());
        printf("%f\n", rep.wrmserror);

        // print out the fitting procedure
        /*for (int i = 0; i < y.size(); i++){
            printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - arctangent(c[0], c[1], x[i][0]));
        }*/

        // nonlinear square curve fitting for Gudermannian function
        lsfitcreatewfg(x, y, w, c, state);
        lsfitsetcond(state, epsx, maxits);
        alglib::lsfitfit(state, gudermannian_f, gudermannian_fd);
        lsfitresults(state, c, rep);
        //printf("%d\n", int(rep.terminationtype));
        printf("%s\n", c.tostring(1).c_str());
        printf("%f\n", rep.wrmserror);

        // print out the fitting procedure
        /*for (int i = 0; i < y.size(); i++){
            printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - gudermannian(c[0], c[1], x[i][0]));
        }*/
        
        // nonlinear square curve fitting for simple algebraic function
        lsfitcreatewfg(x, y, w, c, state);
        lsfitsetcond(state, epsx, maxits);
        alglib::lsfitfit(state, algebraic_f, algebraic_fd);
        lsfitresults(state, c, rep);
        //printf("%d\n", int(rep.terminationtype));
        printf("%s\n", c.tostring(1).c_str());
        printf("%f\n", rep.wrmserror);

        // print out the fitting procedure
        /*for (int i = 0; i < y.size(); i++){
            printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - algebraic(c[0], c[1], x[i][0]));
        }*/
    
    } catch(alglib::ap_error alglib_exception){
        printf("ALGLIB exception with message '%s'\n", alglib_exception.msg.c_str());
        return 1;
    }

    return 0;
}