
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

//sigmoud activation function
double sigmoid(double x) {
    double result;
    result = 1 / (1 + exp(-x));
    return result;
}

//derivative of the sigmoid activation function
double Dsigmoid(double x) {
    double result;
    result = exp(-x) / ((1 + exp(-x))* (1 + exp(-x)));
    return result;
}

//reLu activation function
double reLu(double x) {
    double result;
    if (x >= 0) {result = x;}
    if (x < 0){result = 0;}
    return result;
}

//derivative of the activation function (by convention I set the derivative at 0 to be one even though it is undefined)
double DreLu(double x) {
    double result;
    if (x >= 0) {result = 1;}
    if (x < 0){result = 0;}    
    return result;
}
