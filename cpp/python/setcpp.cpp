#ifndef SETPCPP_HPP
#define SETPCPP_HPP

#include <pybind11/pybind11.h>
#include <setcpp.h>

PYBIND11_MODULE(setcpp, m) {

    m.def("euclidean_cluster", &euclidean_cluster, "Segmentation with Euclidean_cluster");
    m.def("enforce_smoothness", &enforceSmoothness, "Enforce smoothness of ranges");

    py::class_<SmoothnessDPL1Cost, std::shared_ptr<SmoothnessDPL1Cost>>(m, "SmoothnessDPL1Cost")
        .def(py::init<int, float, float, float> ())
        .def("smoothedRanges", &SmoothnessDPL1Cost::smoothedRanges)
        .def("randomCurtain", &SmoothnessDPL1Cost::randomCurtain)
        ;

    py::class_<SmoothnessDPPairGridCost, std::shared_ptr<SmoothnessDPPairGridCost>>(m, "SmoothnessDPPairGridCost")
        .def(py::init<int, float, float, float> ())
        .def("getRanges", &SmoothnessDPPairGridCost::getRanges)
        .def("smoothedRanges", &SmoothnessDPPairGridCost::smoothedRanges)
        .def("randomCurtain", &SmoothnessDPPairGridCost::randomCurtain)
        ;

    py::class_<SmoothnessGreedy, std::shared_ptr<SmoothnessGreedy>>(m, "SmoothnessGreedy")
            .def(py::init<int, float, float, float> ())
            .def("smoothedRanges", &SmoothnessGreedy::smoothedRanges)
            ;

    #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
    #else
    m.attr("__version__") = "dev";
    #endif
}

#endif