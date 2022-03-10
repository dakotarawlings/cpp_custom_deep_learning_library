cmake_minimum_required(VERSION 3.4...3.18)
project(test LANGUAGES CXX)

add_subdirectory(pybind11)


pybind11_add_module(FFNN_pymodule FFNN_module.cpp)