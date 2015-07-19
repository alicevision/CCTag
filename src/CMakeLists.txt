FILE( GLOB_RECURSE
      CCTag_hpp
      ./cctag/*.hpp
)

FILE( GLOB_RECURSE
      CCTag_cpp
      ./cctag/*.cpp
)

IF(USE_CUDA)
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
    set_source_files_properties( ${CCTag_cpp}
                                 PROPERTIES COMPILE_FLAGS "-std=c++11"
    )
    # set(CUDA_NVCC_DEBUG_FLAGS "-G -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52")
    # set(CUDA_NVCC_RELEASE_FLAGS "-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 -O3")

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
ELSE(USE_CUDA)
    SET(CUDA_cpp "")

    add_library( CCTag SHARED ${CCTag_cpp} )
    #add_library( CCTagStatic STATIC ${CCTag_cpp} )
ENDIF(USE_CUDA)

SET(CCTAG_LINK_LIBRARIES
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
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

target_link_libraries( CCTag PUBLIC ${CCTAG_LINK_LIBRARIES} )
#target_link_libraries( CCTagStatic PUBLIC ${CCTAG_LINK_LIBRARIES} )

target_include_directories( CCTag PUBLIC .)

set(CCTagDetect_cpp ./cctag/detection/main.cpp)

if(USE_CUDA)
    set_source_files_properties( ${CCTagDetect_cpp}
                                 PROPERTIES COMPILE_FLAGS "-std=c++11"
    )
    add_executable( detection ${CCTagDetect_cpp} ${cuda_files} )
    target_link_libraries( detection CCTag
                                     ${CUDA_LIBRARIES}
                                     ${OPTPP_LIBRARIES}
				     ${Ceres_LIBRARIES}
				     )
else(USE_CUDA)
    add_executable( detection ${CCTagDetect_cpp})
    target_link_libraries( detection CCTag
                                     ${OPTPP_LIBRARIES}
				     ${Ceres_LIBRARIES}
				     )
endif(USE_CUDA)

add_executable( pgmdiff playground/pgmdiff.cc )
target_link_libraries( pgmdiff PUBLIC ${Boost_LIBRARIES} )

install(TARGETS CCTag LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

set( FILES_TO_DEPLOY
      ./cctag/view.hpp
      ./cctag/ICCTag.hpp
      ./cctag/params.hpp
      ./cctag/progBase/MemoryPool.hpp
      ./cctag/progBase/IMemoryPool.hpp
      ./cctag/progBase/pattern/Singleton.hpp)

foreach ( file ${FILES_TO_DEPLOY} )
get_filename_component( dir ${file} DIRECTORY )
install( FILES ${file} DESTINATION include/${dir} )
endforeach()
