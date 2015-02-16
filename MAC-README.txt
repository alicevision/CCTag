Compiling for MacOS X, Mavericks

= Using the native XCode CLang compiler =

This approach is inspired by user Marco's comment on
http://stackoverflow.com/questions/19626854/opencv-cuda-osx-maverick

Step 1:
 - uninstall everything you have form MacPorts
   sudo port uninstall installed
 - re-install gmake and cmake
   sudo port install gmake cmake

Step 2:
 - install the latest version of CUDA, 6.5
 - add CUDA to the path
   export PATH=$PATH:/Developer/NVIDIA/CUDA-6.5/bin
   export CUDA_HOST_COMPILER=/usr/bin/clang
   export CUDA_NVCC_FLAGS="$(CUDA_NVCC_FLAGS) -Xcompiler -stdlib=libstdc++; -Xlinker -stdlib=libstdc++"
   export CMAKE_CXX_FLAGS="$(CMAKE_CXX_FLAGS) -stdlib=libstdc++"
   export CMAKE_EXE_LINKER_FLAGS="$(CMAKE_EXE_LINKER_FLAGS) -stdlib=libstdc++"

Step 3
 - install very simple libraries
 - glog
   mkdir BUILD
   cd BUILD
   LDFLAGS="-stdlib=libstdc++" CXXFLAGS="-stdlib=libstdc++" ../configure --prefix=/opt/local/stdcxx
   make
   sudo make install
 - gsl
   mkdir BUILD
   cd BUILD
   LDFLAGS="-stdlib=libstdc++" CXXFLAGS="-stdlib=libstdc++" ../configure --prefix=/opt/local/stdcxx
   make
   sudo make install
 - jpeg9a
   mkdir BUILD
   cd BUILD
   LDFLAGS="-stdlib=libstdc++" CXXFLAGS="-stdlib=libstdc++" ../configure --prefix=/opt/local/stdcxx
   make
   sudo make install
 - libpng
   mkdir BUILD
   cd BUILD
   LDFLAGS="-stdlib=libstdc++" CXXFLAGS="-stdlib=libstdc++" ../configure --prefix=/opt/local/stdcxx
   make
   sudo make install
 - SuiteSparse
   cd SuiteSparse/SuiteSparse_config
   edit SuiteSparse_config_Mac.mk
     LIB = -lm                       ==>  LIB = -lm -stdlib=libstdc++
     <null>                          ==> CC = clang
     <null>			     ==> CXX = clang
     CFLAGS =                        ==>  CFLAGS = -stdlib=libstdc++
     # MAKE = gmake                  ==> MAKE = gmake
     # CHOLMOD_CONFIG = -DNPARTITION ==> CHOLMOD_CONFIG = -DNPARTITION
     INSTALL_LIB = /usr/local/lib    ==> INSTALL_LIB = /opt/local/stdcxx/lib
     INSTALL_INCLUDE = /usr/local/include ==> INSTALL_INCLUDE = /opt/local/stdcxx/include
   mv SuiteSparse_config_Mac.mk SuiteSparse_config.mk
   cd ..
   make
   sudo make install
 - eigen
   cmake -DCMAKE_INSTALL_PREFIX=/opt/local/stdcxx \
         -DCMAKE_CXX_FLAGS="-stdlib=libstdc++" \
         -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libstdc++" \
	 -DCMAKE_INCLUDE_PATH="/opt/local/stdcxx/include" \
	 -DCMAKE_LIBRARY_PATH="/opt/local/stdcxx/lib" \
         ..

Step 4
 - boost
   (1) configure step
   ./bootstrap.sh cxxflags="-stdlib=libstdc++" \
                  --prefix=/opt/local/stdcxx \
                  --without-libraries=python \
                  --without-libraries=mpi

   (2) post-configure patch, may fix the RPATH defect
   sed -e 's|-install_name \"|&/opt/local/stdcxx/lib/|' tools/build/src/tools/darwin.jam > darwin.jam-2
   mv darwin.jam-2 tools/build/src/tools/darwin.jam

   Note: MacPorts create a user-config.jam file as follows:
   	 echo "using darwin : : ${configure.cxx} : <cxxflags>\"${configure.cxxflags} ${cxx_stdlibflags}\" ${compileflags} <linkflags>\"${configure.ldflags} ${cxx_stdlibflags}\" : ;" > user-config.jam
	 I believe these are covered by the command line flags below.
	 If it is required, add -user-config=user-config.jam as flag below after --layout.

   (3) compile and install
   sudo ./b2 --layout=tagged \
             toolset=darwin \
	     variant=release \
	     link=static,shared \
	     runtime-link=shared \
	     threading=single,multi \
             cxxflags="-stdlib=libstdc++" \
             linkflags="-stdlib=libstdc++" \
	     install --prefix=/opt/local/stdcxx

   Note: online recommendation to add -std=c++11 to the cxxflags leads to lots of
         errors with boost_1_57_0 and clang 6.0 on Mavericks, so I dropped it

 - add boost-gil-numeric
   sudo cp -r numeric /opt/local/stdcxx/include/boost/gil/extension/
 - add boost-numeric-bindings
   git clone http://git.tiker.net/trees/boost-numeric-bindings.git
   ./configure --prefix=/opt/local/stdcxx
   sudo make install

Step 5
 - Ceres
 - call
    cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
          -DCMAKE_C_COMPILER=/usr/bin/gcc \
	  -DCMAKE_INSTALL_PREFIX=/opt/local/stdcxx \
	  -DCMAKE_CXX_FLAGS="-stdlib=libstdc++" \
	  -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libstdc++" \
	  -DCMAKE_INCLUDE_PATH="/opt/local/stdcxx/include" \
	  -DCMAKE_LIBRARY_PATH="/opt/local/stdcxx/lib" \
	  ..

Step 6
 - optpp-2.4
   mkdir BUILD; cd BUILD
   CPPFLAGS="-I/opt/local/stdcxx/include" LDFLAGS="-L/opt/local/stdcxx/lib" CXXFLAGS="-stdlib=libstdc++" LDFLAGS="-stdlib=libstdc++" ../configure --prefix=/opt/local/stdcxx
   make
   sudo make install
   sudo cp include/OPT++_config.h /opt/local/stdcxx/include/

Step 7:
 - opencv version 2.x.x
   (1) OpenCV CMake files screw up RPATH on Mac with cmake 3
   patch -p1 << ../CCTag/OpenCV-2.4.10-mac-fixes.patch

   (2) call cmake and build
   cmake -DCMAKE_INSTALL_PREFIX=/opt/local/stdcxx \
         -DCMAKE_CXX_FLAGS="-stdlib=libstdc++" \
         -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libstdc++" \
         -DCMAKE_INCLUDE_PATH="/opt/local/stdcxx/include" \
         -DCMAKE_LIBRARY_PATH="/opt/local/stdcxx/lib" \
         -DCUDA_ARCH_BIN="2.0 2.1(2.0) 3.0 3.5" \
	 -DBUILD_JPEG=OFF \
	 -DBUILD_ZLIB=OFF \
	 -DBUILD_PNG=OFF \
	 -DBUILD_SHARED_LIBS=ON \
	 -DWITH_OPENCL=OFF \
         ..
   Note: Opencv compiles its own libpng and libjpeg if they don't exist in MacPorts
   versions. That is a bad thing because CCTag will have version collisions.

Step 8
 - compile CCTag
   cmake \
	 -DOpenCV_DIR=/opt/local/stdcxx/share/OpenCV \
         -DCMAKE_INSTALL_PREFIX=/opt/local/stdcxx \
         -DCMAKE_CXX_FLAGS="-stdlib=libstdc++" \
         -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libstdc++ -lstdc++" \
         -DCMAKE_INCLUDE_PATH="/opt/local/stdcxx/include" \
         -DCMAKE_LIBRARY_PATH="/opt/local/stdcxx/lib" \
         -DOpenMP_CXX_FLAGS="-fopenmp" \
         -DOpenMP_C_FLAGS="-fopenmp" \
         ..

