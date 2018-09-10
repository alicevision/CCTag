ARG CUDA_TAG=9.2-devel
FROM nvidia/cuda:$CUDA_TAG
LABEL maintainer="AliceVision Team alicevision@googlegroups.com"

# use CUDA_TAG to select the image version to use
# see https://hub.docker.com/r/nvidia/cuda/
#
# For example to create a ubuntu 16.04 with cuda 8.0 for development, use
# CUDA_TAG=8.0-devel
# docker build --build-arg CUDA_TAG=$CUDA_TAG --tag cctag:$CUDA_TAG .
#
# then execute with nvidia docker (https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))
# docker run -it --runtime=nvidia cctag

# OS/Version (FILE): cat /etc/issue.net
# Cuda version (ENV): $CUDA_VERSION

# System update
RUN apt-get clean && apt-get update && apt-get install -y --no-install-recommends \
                build-essential \
                cmake \
                git \
                wget \
                unzip \
                yasm \
                pkg-config \
                libtool \
                nasm \
                automake \
                libpng12-dev \
                libjpeg-turbo8-dev \
                libboost-all-dev \
                ffmpeg \
                libavcodec-dev \
                libavformat-dev \
                libswscale-dev \
                libeigen3-dev \
                libavresample-dev \
                libtbb-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
ENV OPENCV_VERSION="3.4.1"
RUN wget https://github.com/opencv/opencv/archive/"${OPENCV_VERSION}".zip \
&& unzip ${OPENCV_VERSION}.zip 

WORKDIR /opt/opencv-${OPENCV_VERSION}/cmake_binary \
RUN cmake -j -DBUILD_TIFF=ON \
      -DBUILD_opencv_java=OFF \
      -DWITH_CUDA=OFF \
      -DENABLE_AVX=ON \
      -DWITH_OPENGL=ON \
      -DWITH_IPP=ON \
      -DWITH_TBB=ON \
      -DWITH_EIGEN=ON \
      -DWITH_V4L=ON \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=RELEASE  .. \
    && make -j install \
    && rm /opt/${OPENCV_VERSION}.zip \
    && rm -r /opt/opencv-${OPENCV_VERSION}


COPY . /opt/cctag
WORKDIR /opt/cctag/build
RUN cmake .. -DWITH_CUDA:BOOL=ON \
       -DCMAKE_BUILD_TYPE=Release \
       -DOpenCV_DIR:PATH=/usr/local/share/OpenCV && make install -j
