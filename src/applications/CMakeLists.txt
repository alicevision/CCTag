cmake_minimum_required(VERSION 3.4)

include(ExternalProject)

project(CCTagApplications LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)

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
if(NOT Boost_FOUND)
find_package(Boost 1.53.0 REQUIRED
    COMPONENTS date_time chrono thread serialization system filesystem atomic program_options timer QUIET)
else()
message(STATUS "BOOST already there")
endif()

# OPENCV
find_package(OpenCV REQUIRED core videoio imgproc imgcodecs highgui)

# TBB
if(NOT TBB_FOUND)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
  find_package(TBB 4 REQUIRED)
else()
message(STATUS "TBB already there")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CCTagDetect_cpp ./detection/main.cpp ./detection/CmdLine.cpp)
#set_source_files_properties(${CCTagDetect_cpp} PROPERTIES COMPILE_FLAGS "-std=c++11")

set(CCTagRegression_cpp
  ./regression/main.cpp
  ./regression/TestLog.cpp
  ./regression/Regression.cpp)
#set_source_files_properties(${CCTagRegression_cpp} PROPERTIES COMPILE_FLAGS "-std=c++11")

set(CCTagSimulation_cpp ./simulation/main.cpp)
#set_source_files_properties(${CCTagSimulation_cpp} PROPERTIES COMPILE_FLAGS "-std=c++11")

get_target_property(testprop CCTag::CCTag INTERFACE_INCLUDE_DIRECTORIES )

add_executable(detection ${CCTagDetect_cpp})
target_include_directories(detection PUBLIC ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TBB_INCLUDE_DIR})
target_link_libraries(detection PUBLIC CCTag::CCTag ${TBB_tbb_LIBRARY_RELEASE} ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(regression ${CCTagRegression_cpp})
target_include_directories(regression PUBLIC ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(regression PUBLIC CCTag::CCTag ${OpenCV_LIBS} ${Boost_LIBRARIES} ) 


add_executable(simulation ${CCTagSimulation_cpp})
target_include_directories(simulation PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(simulation PUBLIC ${OpenCV_LIBS})

install(TARGETS detection regression simulation DESTINATION bin)