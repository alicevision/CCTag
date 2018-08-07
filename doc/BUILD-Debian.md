1. install boost (apt-get)
2. install NVidia drivers
3. install CUDA
4. install ccmake
5. install OpenCV
6. install eigen
7. install jpeg9a
8. install libpng
9. install ncurses
10. install CMake
11. create the machine-specific config file
12. compile CCTag

1. install boost (apt-get)
apt install boost

2. Install NVidia drivers
Download the latest drivers from NVidia. Allow them to disable nouveau.
Reboot. Install. Reboot.

3. install CUDA
We need at least CUDA 7.0. Not yet available in debian packages.
Download the latest CUDA version from NVidia. At the time of writing, this
is 7.5. I use the "run" package.
Debian is unsupported. Accpet the license and say Y to unsupported install.
Do not accept the driver, the one install in (3) is probably newer.
Install CUDA to /usr/local/cuda-<version> and allow the link to /usr/local/cuda
Install the samples in an arbitrary place (/home/labo/Packages)

4. install ccmake
apt-get install cmake-curses-gui

5. install OpenCV
We need at least OpenCV 3.0. Not yet available in debian packages.
Download from http://opencv.org/downloads.html
The CMakeLists.txt of OpenCV screws up install paths (insists on /usr/local)
cmake ..
ccmake ..
scroll down to INSTALL_PREFIX and change to your choice (e.g. /home/labo/Install)

6. install eigen
apt-get install libeigen3-dev

7. install jpeg9a
apt-get install libjpeg-dev

8. install libpng
apt-get install libpng12-dev

9. install ncurses
apt-get install libncurses5-dev

10. install CMake
We need at least CMake 3.4 because CUDA Dynamic Parallelism does not
work with CMake before this. Not yet available in debian packages.
Download from https://cmake.org/download/
Configure, compile and install the one that is already installed.

11. create the machine-specific config file
cp mpg-2014-20.cmake.example <hostname>.cmake
change variables in this file, e.g. set Release/Debug

12. compile CCTag
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
