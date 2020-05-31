ARG CUDA_TAG=10.2
ARG OS_TAG=18.04
FROM alicevision/cctag-deps:cuda${CUDA_TAG}-ubuntu${OS_TAG}
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


COPY . /opt/cctag
WORKDIR /opt/cctag/build
RUN cmake .. -DCCTAG_WITH_CUDA:BOOL=ON \
       -DCMAKE_BUILD_TYPE=Release \
       -DBUILD_SHARED_LIBS:BOOL=ON \
       -DCMAKE_PREFIX_PATH:PATH=/opt/ && make install -j$(nproc)
