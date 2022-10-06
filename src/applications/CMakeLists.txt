if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  # I am top-level project, i.e. I am not being include by another project
  cmake_minimum_required(VERSION 3.13)
  project(CCTagApplications LANGUAGES CXX VERSION 1.0.0)
  include(GNUInstallDirs)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# if this is used as a stand-alone project we need to tell whether to use PIC
if(NOT DEFINED BUILD_SHARED_LIBS)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})
endif()

# enable -fPIE for executables when -fpic
# https://cmake.org/cmake/help/v3.17/policy/CMP0083.html
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.14)
  cmake_policy(SET CMP0083 NEW)
  include(CheckPIESupported)
  check_pie_supported()
elseif(CMAKE_POSITION_INDEPENDENT_CODE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # manually add the link flag for gcc
  list(APPEND CMAKE_EXE_LINKER_FLAGS "-pie")
endif()

#Find needed dependencies
# CCTAG
if(TARGET CCTag)
  add_library(CCTag::CCTag ALIAS CCTag)
  message(STATUS "CCTAG already there")
else()
  # Add NO_CMAKE_BUILDS_PATH for windows if using CMake-GUI to build packages
  # to avoid searching in temporary build directory of Foo project
  # See 5:
  #    * http://www.cmake.org/cmake/help/v3.0/command/find_package.html
  find_package(CCTag CONFIG REQUIRED)
endif()

# BOOST
if(NOT TARGET Boost::boost)
  set(BOOST_REQUIRED_COMPONENTS "date_time;chrono;thread;serialization;system;filesystem;atomic;program_options;timer")
  if(WIN32)
    set(BOOST_REQUIRED_COMPONENTS "${BOOST_REQUIRED_COMPONENTS};stacktrace_windbg")
  else()
    set(BOOST_REQUIRED_COMPONENTS "${BOOST_REQUIRED_COMPONENTS};stacktrace_basic")
  endif()

  find_package(Boost 1.66.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS} QUIET)
else()
  message(STATUS "BOOST already there")
endif()

# OPENCV
find_package(OpenCV REQUIRED core videoio imgproc imgcodecs highgui)

# TBB
if(NOT TBB_FOUND)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
  find_package(TBB REQUIRED)
else()
  message(STATUS "TBB already there")
endif()

set(CCTagRegression_cpp
  ./regression/main.cpp
  ./regression/TestLog.cpp
  ./regression/Regression.cpp)

set(CCTagSimulation_cpp
  ./simulation/main.cpp)

get_target_property(testprop CCTag::CCTag INTERFACE_INCLUDE_DIRECTORIES)

set(CCTagDetect_cpp ./detection/main.cpp ./detection/CmdLine.cpp)
add_executable(detection ${CCTagDetect_cpp})

find_package(DevIL COMPONENTS IL ILU) # yields IL_FOUND, IL_LIBRARIES, IL_INCLUDE_DIR

target_include_directories(detection PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
  )
target_link_libraries(detection PUBLIC
  CCTag::CCTag
  TBB::tbb
  ${OpenCV_LIBS}
  Boost::filesystem Boost::program_options Boost::timer
  )

if(IL_FOUND OR DevIL_FOUND)
  message(STATUS "DevIL found")
  target_compile_options(detection PRIVATE -DUSE_DEVIL)
  target_include_directories(detection PUBLIC
    ${IL_INCLUDE_DIR}
    )
  target_link_libraries(detection PUBLIC
    ${IL_LIBRARIES}
    ${ILU_LIBRARIES}
    )
else()
  message(STATUS "DevIL not found")
endif()

add_executable(regression ${CCTagRegression_cpp})
target_include_directories(regression PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(regression PUBLIC
  CCTag::CCTag
  ${OpenCV_LIBS}
  Boost::program_options Boost::serialization
  )

add_executable(simulation ${CCTagSimulation_cpp})
target_include_directories(simulation PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(simulation PUBLIC ${OpenCV_LIBS})

install(TARGETS detection regression simulation DESTINATION ${CMAKE_INSTALL_BINDIR})
