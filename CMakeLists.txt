cmake_minimum_required(VERSION 2.8)
include(ExternalProject)

project( CCTag )

# set the path where we can find the findXXX.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# avoid the cmake policy warning about @rpath in MacOSX
cmake_policy(SET CMP0042 NEW)

# FIND BoostNumericBindings
find_package(BoostNumericBindings REQUIRED)
include_directories(${BoostNumericBindings_INCLUDE_DIRS})
link_directories(${BoostNumericBindings_LIB_DIR})
add_definitions(${BoostNumericBindings_DEFINITIONS})

# FIND UBLAS
find_package(UBlas REQUIRED)
include_directories(${UBlas_INCLUDE_DIRS})
link_directories(${UBlas_LIB_DIR})
add_definitions(${UBlas_DEFINITIONS})
add_definitions(-DBOOST_UBLAS_SHALLOW_ARRAY_ADAPTOR)

# FIND OPENCV
list(APPEND CMAKE_MODULE_PATH "${OpenCV_DIR}")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})
add_definitions(${OpenCV_DEFINITIONS})

# FIND BOOST
# TODO: determine minimal version and minimal components
find_package(Boost 1.53.0 REQUIRED
    COMPONENTS date_time chrono thread serialization system filesystem atomic program_options timer QUIET)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIB_DIR})
add_definitions(${Boost_DEFINITIONS})

# FIND OPTPP
find_package(OPTPP REQUIRED)
include_directories(${OPTPP_INCLUDE_DIRS})
link_directories(${OPTPP_LIB_DIR})
add_definitions(${OPTPP_DEFINITIONS})

# FIND CERES
find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIR})
# link_directories(${Ceres_LIB_DIR}) this variable is not created by Cmake file
add_definitions(${Ceres_DEFINITIONS})

# FIND Eigen
find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIR})
link_directories(${Eigen_LIB_DIR})
add_definitions(${Eigen_DEFINITIONS})

# FIND LAPACK
find_package(LAPACK REQUIRED)

# FIND PNG
find_package(PNG REQUIRED)

# FIND JPEG
find_package(JPEG REQUIRED)

# FIND JPEG
find_package(Glog REQUIRED)

# FIND GOMP
# find_package(Gomp REQUIRED)
# griff: We don't have to find libgomp. Just pass the right flags to gcc
find_package(OpenMP REQUIRED)

# FIND GSL
find_package(GSL REQUIRED)

add_subdirectory(src)

# FIND CMINPACK
#find_package(CMINPACK REQUIRED)
#include_directories(${CMINPACK_INCLUDE_DIRS})
#link_directories(${CMINPACK_LIB_DIR})
#add_definitions(${CMINPACK_DEFINITIONS})
#MESSAGE( "${CMINPACK_DEFINITIONS}" )

#if(NOT CMINPACK_FOUND)
#    message(ERROR " CMINPACK not found!")
#else()
#    message( "-- Found CMINPACK version: ${CMINPACK_VERSION}" )
#endif(NOT CMINPACK_FOUND)
