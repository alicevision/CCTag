TODO

Some examples:
./configure_user -DCMAKE_BUILD_TYPE=Debug
./configure_user -DCMAKE_BUILD_TYPE=Release -DCCTAG_SERIALIZE=1 -DCCTAG_NO_COUT=0

configure_user:

#! /bin/bash
rm -rf ./build
./configure -DCMAKE_CXX_FLAGS_DEBUG="-pg" -DCMAKE_C_FLAGS_DEBUG="-pg" -DOpenCV_DIR=/home/lilian/cpp_workspace/popartExtern/opencv-2.4.9/build -DBoostNumericBindings_DIR=/home/lilian/cpp_workspace/romExtern/boost-numeric-bindings -DOPTPP_ROOT=/home/lilian/cpp_workspace/romExtern/optpp-2.4 --debug-trycompile -DCeres_ROOT_DIR=/home/lilian/cpp_workspace/romExtern/ceres-solver-1.9.0/build $@


