#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>

// a.Logistic function
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = (1.0 / (1.0 + exp(-x[0])) + 1)/2;
}
void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr) {
    // ((x - a) * e^(a + x))/(2(e^a*c + e^c*x)^2)
    grad[0] = ((x-c[1]) * exp(c[0] * (c[1] + x))) / (2 * (((exp(c[0]) * c[1]) + (exp(c[0]) * c[1]))^2));
    grad[1] = -((c[0] * e^(c[0] * (a + x))) / (2(e^(c[1]*c[0]) + e^(c[0]*x))^2));
}

// b.Hyperbolic tangent
void Hyperbolic_tangent_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    // This calculates f(c,x) = (exp(x) - exp(-x))/((exp(x) + exp(-x)))
    func = (exp(x[0]) - exp(-x[0]))/(exp(x[0]) + exp(-x[0]));
}

void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    // This calculates f(c,x) = 1 - ((exp(x) - exp(-x))^2/((exp(x) + exp(-x))^2)/2
   // x[0] is for x
    grad[0] = 1 - (exp(x[0]) - exp(-x[0])) * (exp(x[0]) - exp(-x[0]))/((exp(x[0]) + exp(-x[0])) * (exp(x[0]) + exp(-x[0])))/2;
}

// c.Arctangent function
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = atan(x[0]);
}

void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    grad[0] = (1.0 + 1/(x[0]^2 + 1))/2;
}

// d.Gudermannian function //Todo
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    // This callback calculates f(c,x) = 2 * atan(tanh(c0 * x0 / 2))
    // where x is a position on the X-axis and c is an adjustable parameter.
    func = 2.0 * atanta(tanh(c[0] * x[0] / 2.0));
}

void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    // This callback calculates f(c,x) = 2 * atanh(tanh(c0 * x0 / 2)) and gradient G={df/dc[i]}
    // where x is a position on the X-axis and c is an adjustable parameter.
    // IMPORTANT: the gradient is calculated with respect to C, not to X.
    double gudermannian = 2.0 * atanh(tanh(c[0] * x[0] / 2.0));
    func = gudermannian;
    grad[0] = (1.0 - pow(tanh(c[0] * x[0] / 2.0), 2)) * x[0] / (1.0 - pow(x[0], 2) / 4.0);
}

// e.Error function
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

