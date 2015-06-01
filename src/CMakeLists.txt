FILE( GLOB_RECURSE
      CCTag_hpp
      ./cctag/*.hpp
)

FILE( GLOB_RECURSE
      CCTag_cpp
      ./cctag/*.cpp
)

IF(CUDA_FOUND)
    FILE( GLOB_RECURSE
          CUDA_cpp
          ./cuda/*.cpp
    )
    FILE( GLOB_RECURSE
          CUDA_cu
          ./cuda/*.cu
    )
    set_source_files_properties( ${CUDA_cpp} ${CUDA_cu}
                                 PROPERTIES COMPILE_FLAGS "-g -Wall"
    )
    # set(CUDA_NVCC_FLAGS "-G -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52")
    set(CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 -O3")

    # I try to find parameters that allow Dynamic Parallelism.
    # So far not successful, but the following settings appear
    # to be involved.
    #
    # set(CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 -O3 -rdc=true --compiler-options -fPIC")
    # set(CUDA_SEPARABLE_COMPILATION ON)
    # set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
    # set(BUILD_SHARED_LIBS ON)

    CUDA_COMPILE( cuda_files
                  ${CUDA_cu} )

    add_library( CCTag STATIC ${CUDA_cpp} ${CCTag_cpp} )
ELSE(CUDA_FOUND)
    SET(CUDA_cpp "")

    add_library( CCTag STATIC ${CCTag_cpp} )
ENDIF(CUDA_FOUND)

target_link_libraries( CCTag PUBLIC
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${OPTPP_LIBRARIES}
    ${Ceres_LIBRARIES}
    ${Eigen_LIBRARIES}
    ${PNG_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${Eigen_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${GSL_LIBRARIES}
    ${LAPACK_LIBRARIES}
    pthread
    dl
    ${Gomp_LIBRARIES} )

target_include_directories( CCTag PUBLIC .)

set(CCTagDetect_cpp ./cctag/detection/main.cpp)

if(CUDA_FOUND)
    add_executable( detection ${CCTagDetect_cpp} ${cuda_files} )
    target_link_libraries( detection CCTag ${CUDA_LIBRARIES} )
else(CUDA_FOUND)
    add_executable( detection ${CCTagDetect_cpp})
    target_link_libraries( detection CCTag )
endif(CUDA_FOUND)

install(TARGETS CCTag LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

file(GLOB_RECURSE FILES_TO_DEPLOY ./cctag/*.hpp)
INSTALL( FILES ${FILES_TO_DEPLOY} DESTINATION include/cctag )
INSTALL( DIRECTORY ${DIRS_TO_DEPLOY} DESTINATION include/cctag )
