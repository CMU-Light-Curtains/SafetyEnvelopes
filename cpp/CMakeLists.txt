cmake_minimum_required(VERSION 2.8.3)
project(setcpp_lib)

set (CMAKE_CXX_STANDARD 11)

find_package(PythonLibs 3)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
add_subdirectory(lib/pybind11)

# setcpp
file(GLOB setcpp_SRC
	"include/*.h"
	"src/*.cpp"
)
message("${setcpp_SRC}")

include_directories(
	${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
	include
	python
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(setcpp_lib ${setcpp_SRC})
target_link_libraries(
  setcpp_lib
  pybind11::module
  ${PCL_LIBRARIES}
)
set_target_properties(setcpp_lib PROPERTIES LINKER_LANGUAGE CXX)

# pysetcpp
pybind11_add_module(setcpp python/setcpp.cpp)
target_link_libraries(setcpp PRIVATE pybind11::module setcpp_lib ${PCL_LIBRARIES})
