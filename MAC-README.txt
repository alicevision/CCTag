Compiling for MacOS X, Mavericks

= Using the native XCode CLang compiler =

This approach is inspired by user Marco's comment on
http://stackoverflow.com/questions/19626854/opencv-cuda-osx-maverick

Step 1:
 - uninstall gcc/g++ and gcc_select from MacPorts

Step 2:
 - install the latest version of CUDA, 6.5
 - add CUDA to the path
   export PATH=$PATH:/Developer/NVIDIA/CUDA-6.5/bin
   export CUDA_HOST_COMPILER=/usr/bin/clang
   export CUDA_NVCC_FLAGS="$(CUDA_NVCC_FLAGS) -Xcompiler -stdlib=libstdc++; -Xlinker -stdlib=libstdc++"
   export CMAKE_CXX_FLAGS="$(CMAKE_CXX_FLAGS) -stdlib=libstdc++"
   export CMAKE_EXE_LINKER_FLAGS="$(CMAKE_EXE_LINKER_FLAGS) -stdlib=libstdc++"

Step 3:
 - Goal: install OpenCV such that it links libstdc++ instead of libc++
 - create BUILD directory, cd into it, call "cmake .."
 - switch to advanced mode using "t"
 - add -stdlib=libstdc++ in both CMAKE_CXX_FLAGS and CMAKE_EXE_LINKER_FLAGS
 - change PREFIX to /opt/local
 The following alternative failed:
 - call
   sudo port install -s opencv configure.cxx_stdlib="libstdc++"

Step 4
 - install boost
 - call
   ./bootstrap.sh cxxflags="-stdlib=libstdc++"
   ./b2 cxxflags="-stdlib=libstdc++" linkflags="-stdlib=libstdc++"
   sudo ./b2 cxxflags="-stdlib=libstdc++" linkflags="-stdlib=libstdc++" install --prefix=/opt/local
 - Note: online recommendation to add -std=c++11 to the cxxflags leads to lots of
         errors with boost_1_57_0 and clang 6.0 on Mavericks, so I dropped it
 - add boost-gil-numeric
   sudo cp -r numeric /opt/local/include/boost/gil/extension/

Step 5
 - install Ceres
 - make sure the cmake cache is empty
 - call
   ccmake -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
          -DCMAKE_C_COMPILER=/usr/bin/gcc \
	  -DCMAKE_INSTALL_PREFIX=/opt/local \
	  -DCMAKE_CXX_FLAGS="-stdlib=libstdc++" \
	  -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libstdc++" \
	  ..
 // - call ccmake, configure, go to advanced mode (t)
 // - set CMAKE_INSTALL_PREFIX /opt/local
 // - set CMAKE_CXX_FLAGS -stdlib=libstdc++
 // - set CMAKE_EXE_LINKER_FLAGS -stdlib=libstdc++
 // - set CMAKE_CXX_COMPILER /usr/bin/g++
 // - set CMAKE_C_COMPILER /usr/bin/gcc
 Note: make sure that this g++ is actually the gcc frontend for llvm 6.0
       i.e. that it is call compatible with the clang used for boost
 Note: g++ is needed because it has --with-gxx-include-dir=/usr/include/c++/4.2.1
       which gives it unordered_map and shared_ptr
 Note: I had to remove google::SetUsageMessage from an example

Step 6
 - install optpp-2.4
 CXXFLAGS="-stdlib=libstdc++" LDFLAGS="-stdlib=libstdc++" ./configure --prefix=/opt/local
 make
 sudo make install


| Here's how I compiled OpenCV 2.4.8 on OSX Mavericks 10.9.1 using Xcode 5.0.2 and CUDA 5.5:
| 
| open CMake to set the project, and to the basic configuration
| in latest Xcode (I think >= 5) there's no more the gcc compiler, deprecated in favor of clang, so go to the CUDA options of the CMAKE project and change CUDA_HOST_COMPILER to use "/usr/bin/clang". Luckily CUDA 5.5 supports clang and not only gcc
| Apparently CUDA 5.5 supports only the older libstdc++ library and not the more modern libc++, so update CUDA_NVCC_FLAGS to tell mvcc to pass tell the nativa compilar to use this older library. Add "-Xcompiler -stdlib=libstdc++; -Xlinker -stdlib=libstdc++"
| Tell also the C++ compiler that compiles the rest of the library to use libstdc++: show the advanced options of CMAKE and go to CMAKE to add "-stdlib=libstdc++" to both CMAKE_CXX_FLAGS and CMAKE_EXE_LINKER_FLAGS

