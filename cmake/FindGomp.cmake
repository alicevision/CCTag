find_package(OpenMP REQUIRED)

# griff: CMake own OpenMP package solves the problem for gcc and other compilers
#TODO
if(APPLE)
  message( STATUS "MacOSX: use -fopenmp instead of linking gomp" )
else(APPLE)
  # set(Gomp_LIBRARIES "/usr/lib/gcc/x86_64-linux-gnu/4.7/libgomp.so")
  set(Gomp_LIBRARIES "gomp" )
endif(APPLE)


