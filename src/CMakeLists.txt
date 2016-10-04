# IF(COMMAND cmake_policy)
cmake_policy(SET CMP0054 OLD) # behaviour keywords in CMake if tests
# ENDIF(COMMAND cmake_policy)

file( GLOB_RECURSE
      CCTag_hpp
      ./cctag/*.hpp
)

file( GLOB_RECURSE
      CCTag_cpp
      ./cctag/*.cpp
)

if(USE_CUDA)
    file( GLOB_RECURSE
          CUDA_cpp
          ./cuda/*.cpp
    )
    file( GLOB_RECURSE
          CUDA_cu
          ./cuda/*.cu
    )

    include_directories("${CMAKE_SOURCE_DIR}/src")
    include_directories("${CMAKE_SOURCE_DIR}/3rdparty/Cuda-7.0-cub")

    set_source_files_properties( ${CUDA_cu} PROPERTIES COMPILE_FLAGS "-Wall --std=c++11")
    set_source_files_properties( ${CUDA_cpp} PROPERTIES COMPILE_FLAGS "-Wall --std=c++11")
    set_source_files_properties( ${CCTag_cpp} PROPERTIES COMPILE_FLAGS "-std=c++11")

    add_definitions("-DSM_ARCH=100")
    # add_definitions("-DSM_TARGETS=-gencode=arch=compute_52,code=\"sm_52,compute_52\"")
    add_definitions("-DSM=100")

    # set(CUDA_SEPARABLE_COMPILATION ON)
    # set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
    # set(BUILD_SHARED_LIBS ON)


    if(NOT COMPILE_STATIC_CCTAG_LIBRARY)
    	message( FATAL_ERROR "We must link CCTag library static with CUDA" )
    endif()
    add_library( CCTag STATIC ${CCTag_cpp} )
    CUDA_ADD_LIBRARY( CCTagCuda STATIC ${CCTag_cpp} ${CUDA_cpp} ${CUDA_cu} )
else(USE_CUDA)
    set(CUDA_cpp "")

    if( COMPILE_STATIC_CCTAG_LIBRARY )
    	message( STATUS "Compiling static library" )
    	add_library( CCTag STATIC ${CCTag_cpp} )
    else( COMPILE_STATIC_CCTAG_LIBRARY )
    	message( STATUS "Compiling shared library" )
    	add_library( CCTag SHARED ${CCTag_cpp} )
    endif( COMPILE_STATIC_CCTAG_LIBRARY )
endif(USE_CUDA)

set(CCTAG_LINK_LIBRARIES
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${Eigen_LIBRARIES}
    ${PNG_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${Eigen_LIBRARIES}
    ${TBB_LIBRARIES}
    pthread
    dl )

target_link_libraries(CCTag 
                      PUBLIC ${CCTAG_LINK_LIBRARIES})
target_include_directories(CCTag PUBLIC . ${TBB_INCLUDE_DIR})

install(
  DIRECTORY cctag
  DESTINATION include
  COMPONENT headers
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

if(USE_CUDA)
    install(TARGETS CCTag CCTagCuda LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
else(USE_CUDA)
    install(TARGETS CCTag LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
endif(USE_CUDA)

add_subdirectory(applications)