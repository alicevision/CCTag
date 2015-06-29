if(APPLE)
cmake_minimum_required(VERSION 3.0)
else(APPLE)
cmake_minimum_required(VERSION 2.8)
endif(APPLE)

include(ExternalProject)

project( CCTag )

# load local machine config <hostname>.cmake
site_name(MACHINE_NAME)

include("${MACHINE_NAME}.cmake")

IF(DEFINED CCTAG_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE ${CCTAG_BUILD_TYPE})
ELSE(DEFINED CCTAG_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF(DEFINED CCTAG_BUILD_TYPE)
MESSAGE(STATUS "using built type ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -DTHIS_IS_DEBUG_MODE -pg")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DTHIS_IS_RELEASE_MODE")

# CCTAG_BUILD_MODE set to RELEASE by default (DEBUG,RELEASE)
if(DEFINED CCTAG_BUILD_MODE)
  add_definitions("-D${CCTAG_BUILD_MODE}")
  if(CCTAG_BUILD_MODE STREQUAL "DEBUG")
    add_definitions("-DCCTAG_SERIALIZE")
  endif(CCTAG_BUILD_MODE STREQUAL "DEBUG")
endif(DEFINED CCTAG_BUILD_MODE)

# Store all the output
if(CCTAG_SERIALIZE)
  add_definitions("-DCCTAG_SERIALIZE")
endif(CCTAG_SERIALIZE)

# Store all the output
if(CCTAG_VOTE_DEBUG)
  add_definitions("-DCCTAG_VOTE_DEBUG")
endif(CCTAG_VOTE_DEBUG)

# Disable output stream
if(CCTAG_NO_COUT)
  add_definitions("-DCCTAG_NO_COUT")
endif(CCTAG_NO_COUT)

# set the path where we can find the findXXX.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-extensions")

  # avoid the cmake policy warning about @rpath in MacOSX
  cmake_policy(SET CMP0042 NEW)

  SET(CMAKE_MACOSX_RPATH TRUE) # initialize the MACOSX_RPATH property on all targets
  SET(CMAKE_SKIP_BUILD_RPATH  FALSE) # don't skip the full RPATH for the build tree
  # SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) # when building, don't use the install RPATH already
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) # when building, use the install RPATH already
                                           # probably not needed
  # SET(CMAKE_INSTALL_RPATH "") # the RPATH to be used when installing
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) # LC_RPATH for CUDA and OpenCV etc written into executable
endif(APPLE)

# FIND BOOST
# TODO: determine minimal version and minimal components
find_package(Boost 1.53.0 REQUIRED
    COMPONENTS date_time chrono thread serialization system filesystem atomic program_options timer QUIET)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  # link_directories(${Boost_LIB_DIR})
  add_definitions(${Boost_DEFINITIONS})
  message( STATUS "Found BOOST: ${Boost_LIBRARIES}" )

  # FIND BoostNumericBindings
  find_package(BoostNumericBindings REQUIRED)
  include_directories(${BoostNumericBindings_INCLUDE_DIRS})
  # link_directories(${BoostNumericBindings_LIB_DIR})
  add_definitions(${BoostNumericBindings_DEFINITIONS})
else(Boost_FOUND)
  message( FATAL_ERROR "Boost not found" )
endif(Boost_FOUND)

# No matter if we want to use CUDA or not. It's needed for OpenCV.
find_package(CUDA 7.0)

if(USE_CUDA)
  message( STATUS "Try using CUDA" )
  if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions("-DWITH_CUDA")
    set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG} -DTHIS_IS_NVCC_DEBUG_MODE -G -g")
    set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -DTHIS_IS_NVCC_RELEASE_MODE -O3")
    set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS} -DTHIS_IS_NVCC_FLAGS -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52")
  else(CUDA_FOUND)
    message( FATAL_ERROR "Failed to find CUDA" )
  endif(CUDA_FOUND)
else(USE_CUDA)
  message( STATUS "Avoid using CUDA" )
endif(USE_CUDA)


# FIND UBLAS
find_package(UBlas REQUIRED)
include_directories(${UBlas_INCLUDE_DIRS})
# link_directories(${UBlas_LIB_DIR})
add_definitions(${UBlas_DEFINITIONS})
add_definitions(-DBOOST_UBLAS_SHALLOW_ARRAY_ADAPTOR)

# FIND OPENCV
list(APPEND CMAKE_MODULE_PATH "${OpenCV_DIR}")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# FIND OPTPP
find_package(OPTPP REQUIRED)
include_directories(${OPTPP_INCLUDE_DIRS})
# link_directories(${OPTPP_LIB_DIR})
add_definitions(${OPTPP_DEFINITIONS})

# FIND CERES
find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIR})
# link_directories(${Ceres_LIB_DIR}) this variable is not created by Cmake file
add_definitions(${Ceres_DEFINITIONS})

# FIND Eigen
find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIR})
# link_directories(${Eigen_LIB_DIR})
add_definitions(${Eigen_DEFINITIONS})

# FIND LAPACK
find_package(LAPACK REQUIRED)

# FIND PNG
find_package(PNG REQUIRED)

# FIND JPEG
find_package(JPEG REQUIRED)

# FIND Glog
find_package(Glog REQUIRED)

# FIND GOMP
find_package(Gomp REQUIRED)
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
