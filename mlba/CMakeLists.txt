cmake_minimum_required(VERSION 2.8.12)
project(pymlba)

add_subdirectory(pybind11)
pybind11_add_module(pymlba pymlba.cpp mlba.cpp mlba.hpp)


find_package(GSL REQUIRED)    # See below (2)

target_link_libraries(pymlba PRIVATE pybind11::module GSL::gsl GSL::gslcblas)
  


