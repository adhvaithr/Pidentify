#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <future>
#include <thread>
#include "interpolation.h"
#include "fit.h"
#include "modelState.h"
#include <iterator>

using namespace alglib;

const double PI = 2 * acos(0);

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

// hyperbolic tangent function
double hyperbolic_tangent(double k, double alpha, double x)
{
    return ((exp(k * (x - alpha)) - exp(-k * (x - alpha))) / (exp(k * (x - alpha)) + exp(-k * (x - alpha))) + 1) / 2;
}

void hyperbolic_f(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - hyperbolic_tangent(c[0], c[1], x[0]);
}

void hyperbolic_fd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    func = 1 - hyperbolic_tangent(c[0], c[1], x[0]);
    grad[0] = - (2 * (x[0] - c[1]) * exp(2 * c[0] * (x[0] - c[1]))) / ((exp(2 * c[0] * (x[0] - c[1])) + 1) * (exp(2 * c[0] * (x[0] - c[1])) + 1));
    grad[1] = (2 * c[0] * exp(2 * c[0] * (x[0] - c[1]))) / ((exp(2 * c[0] * (x[0] - c[1])) + 1) * (exp(2 * c[0] * (x[0] - c[1])) + 1));
}

// arctangent function
double arctangent(double k, double alpha, double x)
{
    return (atan(k * (x - alpha)) + PI / 2) / PI;
}

void arctangent_f(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - arctangent(c[0], c[1], x[0]);
}

void arctangent_fd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    func = 1 - arctangent(c[0], c[1], x[0]);
    grad[0] = - ((x[0] - c[1]) / (PI * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1)));
    grad[1] =  c[0] / (PI * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1));
}

// gudermannian function
double gudermannian(double k, double alpha, double x)
{
    return ((2 * atan(tanh(k * (x - alpha)/ 2))) + PI / 2) / PI;
}

void gudermannian_f(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - gudermannian(c[0], c[1], x[0]);
}

void gudermannian_fd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    func = 1 - gudermannian(c[0], c[1], x[0]);
    grad[0] = -((x[0] - c[1]) * sech(1/2 * c[0] * (x[0] - c[1])) * sech(1/2 * c[0] * (x[0] - c[1]))) / (PI * ((tanh(1/2 * c[0] * (x[0] - c[1])) * tanh(1/2 * c[0] * (x[0] - c[1])) + 1)));
    grad[1] = c[0] * sech(1/2 * c[0] * (x[0] - c[1])) * sech(1/2 * c[0] * (x[0] - c[1])) / (PI * (tanh(1/2 * c[0] * (x[0] - c[1])) * tanh(1/2 * c[0] * (x[0] - c[1])) + 1));
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

// -------------------------------------------------------------------------
// GOMPERTZ FUNCTION
// f(x) = exp( -k * exp( -alpha*x ) )
// We'll define f(x)=1 - Gompertz(...) to keep it consistent with the others.
// -------------------------------------------------------------------------
double gompertz(double k, double alpha, double x)
{
    return std::exp(-k * std::exp(-alpha * x));
}

void gompertz_f(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    double k     = c[0];
    double alpha = c[1];
    double val   = gompertz(k, alpha, x[0]); 
    func = 1.0 - val;
}

void gompertz_fd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    double k     = c[0];
    double alpha = c[1];
    double xx    = x[0];
    double val   = gompertz(k, alpha, xx);

    // func = 1 - val
    func = 1.0 - val;
    
    // d/dk of [1 - val] = - d/dk [val]
    // val = exp(-k * exp(-alpha*x))
    // d[val]/d[k] = val * [ - exp(-alpha*x) ]
    // => d/dk(1 - val) = + exp(-alpha*x) * val
    grad[0] = std::exp(-alpha * xx) * val;

    // d/dalpha of [1 - val] = - d/dalpha [val]
    // d[val]/d[alpha] = val * [ -k * d/d[alpha]( exp(-alpha*x) ) ] = val * [ -k * ( -x * exp(-alpha*x) ) ] = + k*x * exp(-alpha*x) * val
    // => d/dalpha(1 - val) = - [ + k*x * exp(-alpha*x) * val ] = -k * x * exp(-alpha*x) * val
    grad[1] = -k * xx * std::exp(-alpha * xx) * val;
}

// error function based sigmoid
double erf_sigmoid(double k, double alpha, double x) {
    // Compute z = k * (x - alpha)
    double z = k * (x - alpha);
    // Sigmoid: 0.5 * (1 + erf(z))
    return 0.5 * (1 + erf(z));
}

// Function evaluation wrapper for ALGLIB.
// Our fitting function is defined as: f(x) = 1 - erf_sigmoid(k, alpha, x)
void erf_sigmoid_f(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr) {
    func = 1 - erf_sigmoid(c[0], c[1], x[0]);
}

// Derivative (gradient) evaluation for ALGLIB.
void erf_sigmoid_fd(const real_1d_array &c, const real_1d_array &x, 
                    double &func, real_1d_array &grad, void *ptr) {
    double k = c[0];
    double alpha = c[1];
    double z = k * (x[0] - alpha);
    
    // Compute the sigmoid value using erf.
    double g = 0.5 * (1 + erf(z));
    // Our fitting function is f(x) = 1 - g(x) = 0.5 * (1 - erf(z))
    func = 1 - g;
    
    // The derivative of erf(z) is (2/sqrt(pi)) * exp(-z^2)
    // So the derivative of f w.r.t z is: df/dz = - (1/sqrt(pi)) * exp(-z*z)
    double common_deriv = - exp(-z*z) / sqrt(PI); // PI is defined as 2*acos(0)
    
    // Using chain rule:
    // dz/dk = (x - alpha) and dz/dalpha = -k
    grad[0] = common_deriv * (x[0] - alpha);  // Partial derivative w.r.t. k
    grad[1] = common_deriv * (-k);            // Partial derivative w.r.t. alpha
}

void fitFunction(alglib::real_2d_array& x, alglib::real_1d_array& y, alglib::real_1d_array& w,
    void (*inverse_f)(const alglib::real_1d_array&, const alglib::real_1d_array&, double&, void*),
    void (*gradient_f) (const alglib::real_1d_array&, const alglib::real_1d_array&, double&, alglib::real_1d_array&, void*),
    const std::string& functionName, std::vector<FitResult>& results) {
    real_1d_array c = "[0.367, 0.45]"; // initial values for c & a in c(x-a)
    double epsx = 0;
    ae_int_t maxits = 0;
    lsfitstate state;
    lsfitreport rep;

    lsfitcreatewfg(x, y, w, c, state);
    lsfitsetcond(state, epsx, maxits);
    alglib::lsfitfit(state, *inverse_f, *gradient_f);
    lsfitresults(state, c, rep);
    results.push_back({ c, functionName, rep.wrmserror });
}

void curveFitting(std::vector<double> sorted_distances, std::vector<double> y_values, std::string className)
{
    alglib::real_2d_array x;
    alglib::real_1d_array y;
    alglib::real_1d_array w;
    std::vector<FitResult> results;

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

    // nonlinear square curve fitting for logistic function
    fitFunction(x, y, w, &logistic_f, &logistic_fd, "Logistic function", results);
    //printf("%d\n", int(rep.terminationtype));  // status code

    // print out the fitting procedure
    /*for (int i = 0; i < y.length(); i++) {
        printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - logistic(c[0], c[1], x[i][0]));
    }*/

    // nonlinear square curve fitting for hyperbolic tangent function
    fitFunction(x, y, w, &hyperbolic_f, &hyperbolic_fd, "hyperbolic tangent function", results);
    //printf("%d\n", int(rep.terminationtype));

    // print out the fitting procedure
    /*for (int i = 0; i < y.length(); i++) {
        printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - hyperbolic_tangent(c[0], c[1], x[i][0]));
    */
    
    // nonlinear square curve fitting for arctangent function
    fitFunction(x, y, w, &arctangent_f, &arctangent_fd, "arctangent function", results);
    //printf("%d\n", int(rep.terminationtype));

    // print out the fitting procedure
    /*for (int i = 0; i < y.size(); i++){
        printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - arctangent(c[0], c[1], x[i][0]));
    }*/
    
    // nonlinear square curve fitting for Gudermannian function
    fitFunction(x, y, w, &gudermannian_f, &gudermannian_fd, "gudermannian function", results);
    //printf("%d\n", int(rep.terminationtype));

    // print out the fitting procedure
    /*for (int i = 0; i < y.size(); i++){
        printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - gudermannian(c[0], c[1], x[i][0]));
    }*/
    
    // nonlinear square curve fitting for simple algebraic function
    fitFunction(x, y, w, &algebraic_f, &algebraic_fd, "simple algebraic function", results);
    //printf("%d\n", int(rep.terminationtype));

    // ----------------------------------------------------------------------
    // nonlinear square curve fitting for Gompertz function
    // ----------------------------------------------------------------------
    fitFunction(x, y, w, &gompertz_f, &gompertz_fd, "Gompertz function", results);

    // Nonlinear squares curve fitting for error function based sigmoid
    fitFunction(x, y, w, &erf_sigmoid_f, &erf_sigmoid_fd, "error function based sigmoid", results);
    

    m.lock();
    std::cout << "Curve fitting for class \"" << className << "\":\n";
    // print out all results
    for (const auto& result : results) {
        std::cout << "Function: " << result.functionName << std::endl;
        std::cout << "c & a in c(x-a): " << result.c.tostring(1).c_str() << std::endl;
        std::cout << "Residual: " << result.wrmsError << std::endl;
    }

    // print out the best result
    FitResult bestFit = results[0];
    for (const auto& result : results) {
        if (result.wrmsError < bestFit.wrmsError) {
            bestFit = result;
        }
    }

    std::cout << "Best fit function: " << bestFit.functionName << std::endl;
    std::cout << "c & a in c(x-a): " << bestFit.c.tostring(1).c_str() << std::endl;
    std::cout << "Residual: " << bestFit.wrmsError << std::endl;

    // Save best fit function
    MODEL_STATE.bestFit[className] = std::move(bestFit);
    m.unlock();
}

int fitClasses(std::unordered_map<std::string, std::vector<double> >& sorted_distances) {
    std::unordered_map<std::string, std::thread> threads;
    std::unordered_map<std::string, std::future<void> > results;

    for (auto& pair : sorted_distances) {
        size_t l = pair.second.size();

        // construct corresponding y values in terms of distances for ECDF points
        std::vector<double> y(l);
        for (size_t i = 0; i < l; ++i) {
            y[i] = 1 - static_cast<double>(i + 1) / (l + 1);
        }

        /*
        // Insert (0,0) and faraway point into ECDF points
        pair.second.insert(pair.second.begin(), 0);
        y[0] = 1;
        pair.second.insert(pair.second.end(), 1);
        y[l + 1] = 0;
        */
        
        std::packaged_task<void(std::vector<double>, std::vector<double>, std::string)> parallelCurveFitting{ curveFitting };
        results[pair.first] = parallelCurveFitting.get_future();
        threads[pair.first] = std::thread{ std::move(parallelCurveFitting), pair.second, y, pair.first };
    }

    for (auto& pair : threads) {
        try {
            pair.second.join();
            results[pair.first].get();
        }
        catch (alglib::ap_error alglib_exception) {
            std::cout << "While curve fitting for class \"" << pair.first << "\", the following exception occurred:\n";
            printf("ALGLIB exception with message '%s'\n", alglib_exception.msg.c_str());
            return 1;
        }
    }
    
    return 0;
}