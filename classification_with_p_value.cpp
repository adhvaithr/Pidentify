#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>

// a.Logistic function
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    // This callback calculates f(c,x) = 1 / (1 + exp(-c0 * x0^2))
    // where x is a position on the X-axis and c is an adjustable parameter.
    func = 1.0 / (1.0 + exp(-c[0] * pow(x[0], 2)));
}

void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    // This callback calculates f(c,x) = 1 / (1 + exp(-c0 * x0^2)) and gradient G={df/dc[i]}
    // where x is a position on the X-axis and c is an adjustable parameter.
    // IMPORTANT: the gradient is calculated with respect to C, not to X.
    double sigma = 1.0 / (1.0 + exp(-c[0] * pow(x[0], 2)));
    func = sigma;
    grad[0] = sigma * (1.0 - sigma) * (-pow(x[0], 2));
}

// b.Arctangent function
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    // This callback calculates f(c,x) = 2 * atan(tanh(c0 * x0 / 2))
    // where x is a position on the X-axis and c is an adjustable parameter.
    func = 2.0 * atan(tanh(c[0] * x[0] / 2.0));
}

void function_cx_1_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    // This callback calculates f(c,x) = 2 * atan(tanh(c0 * x0 / 2)) and gradient G={df/dc[i]}
    // where x is a position on the X-axis and c is an adjustable parameter.
    // IMPORTANT: the gradient is calculated with respect to C, not to X.
    double gudermannian = 2.0 * atan(tanh(c[0] * x[0] / 2.0));
    func = gudermannian;
    grad[0] = 1.0 - pow(tanh(c[0] * x[0] / 2.0), 2);
}

// c.Gudermannian function
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    // This callback calculates f(c,x) = 2 * atan(tanh(c0 * x0 / 2))
    // where x is a position on the X-axis and c is an adjustable parameter.
    func = 2.0 * atanh(tanh(c[0] * x[0] / 2.0));
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

// d.Error function
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

