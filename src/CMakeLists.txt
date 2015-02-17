FILE(
	GLOB_RECURSE
	CCTag_hpp
	./cctag/*.hpp
)

FILE(
	GLOB_RECURSE
	CCTag_cpp
	./cctag/*.cpp
)

set(CCTagDetect_cpp ./cctag/detection/main.cpp)

# add_library( CCTag SHARED ${CCTag_cpp})
add_library( CCTag STATIC ${CCTag_cpp})
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

add_executable( detection ${CCTagDetect_cpp})
target_link_libraries( detection CCTag )

install(TARGETS CCTag LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(FILES ${CCTag_hpp} DESTINATION include/cctag)

