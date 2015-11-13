# IF(COMMAND cmake_policy)
cmake_policy(SET CMP0054 OLD) # behaviour keywords in CMake if tests
# ENDIF(COMMAND cmake_policy)

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

    include_directories("${CMAKE_SOURCE_DIR}/src")
    # include_directories("/home/labo/Install/cub-1.4.1")
    # include_directories("/local/home/griff/GIT/github/cub-1.4.1")
    include_directories("/usr/local/cuda/include/thrust/system/cuda/detail")

    set_source_files_properties( ${CUDA_cpp} ${CUDA_cu}
                                 PROPERTIES COMPILE_FLAGS "-g -Wall"
    )
    set_source_files_properties( ${CCTag_cpp}
                                 PROPERTIES COMPILE_FLAGS "-std=c++11"
    )

    add_definitions("-DSM_ARCH=100")
    # add_definitions("-DSM_TARGETS=-gencode=arch=compute_52,code=\"sm_52,compute_52\"")
    add_definitions("-DSM=100")

    set(CUDA_NVCC_DEBUG_FLAGS "-G -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52")
    set(CUDA_NVCC_RELEASE_FLAGS "-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 -O3")

    # set(CUDA_SEPARABLE_COMPILATION ON)
    # set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
    # set(BUILD_SHARED_LIBS ON)

    # ---- CUDA_COMPILE( cuda_files ${CUDA_cu} )

    IF( COMPILE_STATIC_CCTAG_LIBRARY )
    	ADD_LIBRARY( CCTag STATIC ${CCTag_cpp} )
    	CUDA_ADD_LIBRARY( CCTagCuda STATIC ${CUDA_cpp} ${CUDA_cu} )
    ELSE( COMPILE_STATIC_CCTAG_LIBRARY )
    	MESSAGE( FATAL_ERROR "We must link CCTag library static with CUDA" )
    ENDIF( COMPILE_STATIC_CCTAG_LIBRARY )
ELSE(USE_CUDA)
    SET(CUDA_cpp "")

    IF( COMPILE_STATIC_CCTAG_LIBRARY )
    	MESSAGE( STATUS "Compiling static library" )
    	add_library( CCTag STATIC ${CCTag_cpp} )
    ELSE( COMPILE_STATIC_CCTAG_LIBRARY )
    	MESSAGE( STATUS "Compiling shared library" )
    	add_library( CCTag SHARED ${CCTag_cpp} )
    ENDIF( COMPILE_STATIC_CCTAG_LIBRARY )
ENDIF(USE_CUDA)

SET(CCTAG_LINK_LIBRARIES
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

target_link_libraries( CCTag PUBLIC ${CCTAG_LINK_LIBRARIES} )
target_include_directories( CCTag PUBLIC .)

set(CCTagDetect_cpp ./applications/detection/main.cpp)

if(USE_CUDA)
    set_source_files_properties( ${CCTagDetect_cpp}
                                 PROPERTIES COMPILE_FLAGS "-std=c++11"
    )
    add_executable( detection ${CCTagDetect_cpp} )
    target_link_libraries( detection CCTag
				     CCTagCuda
                                     ${CUDA_LIBRARIES}
                                     ${OPTPP_LIBRARIES}
				     ${Ceres_LIBRARIES}
				     ${CUDA_CUDADEVRT_LIBRARY} )

    CUDA_ADD_EXECUTABLE( testMemoryPool
                         applications/test/testMemoryPool.cpp )
    TARGET_LINK_LIBRARIES( testMemoryPool CCTag
                                          CCTagCuda
					  ${CUDA_LIBRARIES} 
				          ${CUDA_CUDADEVRT_LIBRARY} )

    install(TARGETS CCTag CCTagCuda LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
else(USE_CUDA)
    ADD_EXECUTABLE( detection ${CCTagDetect_cpp} )
    target_link_libraries( detection CCTag
                                     ${OPTPP_LIBRARIES}
				     ${Ceres_LIBRARIES} )
    ADD_EXECUTABLE( testMemoryPool
                    applications/test/testMemoryPool.cpp )
    TARGET_LINK_LIBRARIES( testMemoryPool CCTag )

    install(TARGETS CCTag LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
endif(USE_CUDA)

ADD_EXECUTABLE( pgmdiff playground/pgmdiff.cc )
TARGET_LINK_LIBRARIES( pgmdiff PUBLIC ${Boost_LIBRARIES} )

set( FILES_TO_DEPLOY
      ./cctag/view.hpp
      ./cctag/ICCTag.hpp
      ./cctag/params.hpp
      ./cctag/talk.hpp
      ./cctag/CCTagMarkersBank.hpp
      ./cctag/progBase/MemoryPool.hpp
      ./cctag/progBase/IMemoryPool.hpp
      ./cctag/progBase/pattern/Singleton.hpp)

foreach ( file ${FILES_TO_DEPLOY} )
get_filename_component( dir ${file} DIRECTORY )
install( FILES ${file} DESTINATION include/${dir} )
endforeach()

