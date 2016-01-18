TODO

1. install boost (apt-get)
2. install boost numeric bindings
3. install NVidia drivers
4. install CUDA
5. install ccmake
6. install OpenCV
7. install gfortran
8. install optpp
9. install eigen
10. install glog
11. install SuiteSparse
12. install the Ceres Solver
13. install jpeg9a
14. install libpng
15. install GNU scientific library
16. install ncurses
17. install CMake
18. create the machine-specific config file
19. compile CCTag

2. install boost numeric bindings
Install from GIT. I have installed from tiker.net
	git clone http://git.tiker.net/trees/boost-numeric-bindings.git
Apparently there are other repos at
	http://beta.boost.org/community/sandbox.html
and	https://svn.boost.org/svn/boost/sandbox/numeric_bindings/

3. Install NVidia drivers
Download the latest drivers from NVidia. Allow them to disable nouveau.
Reboot. Install. Reboot.

4. install CUDA
We need at least CUDA 7.0. Not yet available in debian packages.
Download the latest CUDA version from NVidia. At the time of writing, this
is 7.5. I use the "run" package.
Debian is unsupported. Accpet the license and say Y to unsupported install.
Do not accept the driver, the one install in (3) is probably newer.
Install CUDA to /usr/local/cuda-<version> and allow the link to /usr/local/cuda
Install the samples in an arbitrary place (/home/labo/Packages)

5. install ccmake
apt-get install cmake-curses-gui

6. install OpenCV
We need at least OpenCV 3.0. Not yet available in debian packages.
Download from http://opencv.org/downloads.html
The CMakeLists.txt of OpenCV screws up install paths (insists on /usr/local)
cmake ..
ccmake ..
scroll down to INSTALL_PREFIX and change to your choice (e.g. /home/labo/Install)

7. install gfortran
apt-get install gfortran

8. install optpp
Search for optpp on the web. You will probably end up on https://software.sandia.gov/opt++/
We used version 2.4. Unpack and install. Note that one header file must be
copied manually. E.g.:
	../configure --prefix=/home/labo/Install
	make
	make install
	cp include/Opt++_config.h /home/labo/Install/include/

9. install eigen
apt-get install libeigen3-dev

10. install glog
apt-get install libgoogle-glog-dev

11. install SuiteSparse
apt-get install libsuitesparse-dev

12. install the Ceres Solver
Get the latest version.
git clone https://ceres-solver.googlesource.com/ceres-solver
Build and install, e.g.
    cd ceres-solver; mkdir build; cd build
    cmake -DCMAKE_INSTALL_PREFIX=/home/labo/Install ..
    make; make install

13. install jpeg9a
apt-get install libjpeg-dev

14. install libpng
apt-get install libpng12-dev

15. install GNU scientific library
apt-get install libgsl0-dev

16. install ncurses
apt-get install libncurses5-dev

17. install CMake
We need at least CMake 3.4 because CUDA Dynamic Parallelism does not
work with CMake before this. Not yet available in debian packages.
Download from https://cmake.org/download/
Configure, compile and install the one that is already installed.

18. create the machine-specific config file
cp mpg-2014-20.cmake.example <hostname>.cmake
change variables in this file, e.g. set Release/Debug

19. compile CCTag
There is a self-made configure script (this is not an autoconf
configure). It uses cmake to configure in the subdirectory named
build/<current-git-branch>.

./configure \
	-DCMAKE_INSTALL_PREFIX=/home/labo/Install \
	-DOPTPP_ROOT=/home/labo/Install \
	-DCeres_ROOT_DIR=/home/labo/Install
cd build/<current-git-branch>
make
make install
