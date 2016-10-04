#############################################################################

set(CCTagDetect_cpp ./detection/main.cpp ./detection/CmdLine.cpp)

set(CCTagRegression_cpp
  ./regression/main.cpp
  ./regression/TestLog.cpp
  ./regression/Regression.cpp)
set_source_files_properties(${CCTagRegression_cpp} PROPERTIES COMPILE_FLAGS "-std=c++11")

set(CCTagSimulation_cpp ./simulation/main.cpp)
set_source_files_properties(${CCTagSimulation_cpp} PROPERTIES COMPILE_FLAGS "-std=c++11")

if(USE_CUDA)
    set_source_files_properties(${CCTagDetect_cpp} PROPERTIES COMPILE_FLAGS "-std=c++11")
    add_executable( detection ${CCTagDetect_cpp} )
    target_link_libraries(detection
      PUBLIC  CCTag CCTagCuda ${CUDA_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY} ${TBB_tbb_LIBRARY_RELEASE})

    add_executable(regression ${CCTagRegression_cpp})
    target_link_libraries(regression
      PUBLIC CCTag CCTagCuda ${CUDA_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY} ${TBB_tbb_LIBRARY_RELEASE})

else(USE_CUDA)
    add_executable(detection ${CCTagDetect_cpp})
    target_link_libraries(detection PUBLIC CCTag ${TBB_tbb_LIBRARY_RELEASE})

    add_executable(regression ${CCTagRegression_cpp})
    target_link_libraries(regression PUBLIC CCTag ${TBB_tbb_LIBRARY_RELEASE})

endif(USE_CUDA)

add_executable(simulation ${CCTagSimulation_cpp})
target_link_libraries(simulation PUBLIC CCTag ${TBB_tbb_LIBRARY_RELEASE})

#############################################################################
