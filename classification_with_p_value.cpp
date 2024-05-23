#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>

double sech(double x) {
    return 1.0 / std::cosh(x);
}

// a.Logistic function
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    // This calculates 1/(1 + e^-(c * (x - a))
    func = 1.0 / (1.0 + exp(-c[0] * (x[0]-c[1])));
}
void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr) {
    grad[0] = ((x[0]-c[1]) * exp(c[0] * (c[1] - x[0]))) / (exp(c[0] * (c[1] - x[0])) + 1) * (exp(c[0] * (c[1] - x[0])) + 1);
    grad[1] = -(c[0] * exp(c[0] * (c[1] - x[0])) / (exp(c[0] * (c[1] - x[0])) + 1) * (exp(c[0] * (c[1] - x[0]))));
}

// b.Hyperbolic tangent
void Hyperbolic_tangent_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    // This calculates (exp(c*(x-a)) - exp(-c*(x-a)))/((exp(c*(x-a)) + exp(-c*(x-a))))
    // x = c[0] * (x[0] - c[1])
    func = ((exp(c[0] * (x[0] - c[1])) - exp(-c[0] * (x[0] - c[1])))/(exp(c[0] * (x[0] - c[1])) + exp(-c[0] * (x[0] - c[1]))) + 1) / 2;
}

void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    grad[0] = (2 * (x[0] - c[1]) * exp(2 * c[0] * (x[0] - c[1]))) / ((exp(2 * c[0] * (x[0] - c[1])) + 1) *  (exp(2 * c[0] * (x[0] - c[1])) + 1));
    grad[1] = -((2 * c[0] * exp(2 * c[0] * (x[0] - a[0]))) / ((exp(2 * c[0] * (x[0] - a[0])) + 1) * (exp(2 * c[0] * (x[0] - a[0])) + 1)));
}

// c.Arctangent function
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    // this calculates (arctan(c[0] * (x[0] - c[1]) + 1 ) / 2;
    func = (atan(c[0] * (x[0] - c[1])) + 1) / 2;
}

void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    grad[0] = (x[0] - c[1]) / (2 * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1));
    grad[1] = c[0] / (2 * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1));
}

// d.Gudermannian function
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = (2* atan(tanh(c[0] * (x[0] - c[1])/ 2))) + 1) / 2;
}

void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    grad[0] = ((x[0] - c[1]) * sech(1/2 * c[0] * (x[0] - c[1])) * sech(1/2 * c[0] * (x[0] - c[1])))/ (2 * ((tanh(1/2 * c[0] * (x[0] - c[1])) * tanh(1/2 * c[0] * (x[0] - c[1])) + 1));
    grad[1] = -(c[0] * sech(1/2 * c[0] * (x[0] - c[1])) * sech(1/2 * c[0] * (x[0] - c[1])) / (2 * (tanh(1/2 * c[0] * (x[0] - c[1])) * tanh(1/2 * c[0] * (x[0] - c[1])) + 1)));
}

// e.Error function //Todo
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    // This callback calculates f(c,x) = 1 - erf(c0 * x0)
    // where x is a position on the X-axis and c is an adjustable parameter.
    func = 1.0 - erf(c[0] * x[0]);
}

void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    // This callback calculates f(c,x) = 1 - erf(c0 * x0) and gradient G={df/dc[i]}
    // where x is a position on the X-axis and c is an adjustable parameter.
    // IMPORTANT: the gradient is calculated with respect to C, not to X.
    double error_func = erf(c[0] * x[0]);
    func = 1.0 - error_func;
    grad[0] = -2.0 / sqrt(M_PI) * x[0] * exp(-c[0] * c[0] * x[0] * x[0]);
}

// f. A simple algebraic function
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    // This calculate c(x-a)/(sqrt(1 + (c(x-a)^2));
    func = ((c[0] * (x[0] - c[1]) / sqrt(1 + x[0] * x[0])) + 1) / 2;
}

void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    grad[0] = (x[0] - c[1]) / (2 * (sqrt((c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1))));
    grad[1] = c[0] / (2 * (sqrt((c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1))));
}