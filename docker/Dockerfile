FROM nvcr.io/nvidia/deepstream:5.0.1-20.09-devel

LABEL maintainer=Mascom<vuminhduc9755@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive

# Dependece
RUN apt-get update && apt-get install -y --no-install-recommends wget autoconf automake libtool curl make g++ cmake libgtk2.0-0 libtbb2 libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgstrtspserver-1.0-dev libx11-dev


# Build protoc
RUN cd /tmp && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protobuf-all-3.14.0.tar.gz \
    && tar xvzf protobuf-all-3.14.0.tar.gz && cd protobuf-3.14.0/ && ./configure \
    && make -j$(nproc) && make install && ldconfig && cd / && rm -rf /tmp/*

# Opencv
RUN cd / && mkdir installopencv/ && cd installopencv/ && git clone https://github.com/Itseez/opencv.git && git clone https://github.com/Itseez/opencv_contrib.git && cd opencv && mkdir release && cd release && cmake -D BUILD_TIFF=ON -D WITH_CUDA=OFF -D ENABLE_AVX=OFF -D WITH_OPENGL=OFF -D WITH_OPENCL=OFF -D WITH_IPP=OFF -D WITH_TBB=ON -D BUILD_TBB=ON -D WITH_EIGEN=OFF -D WITH_V4L=OFF -D WITH_VTK=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D OPENCV_GENERATE_PKGCONFIG=ON -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/installopencv/opencv_contrib/modules/ /installopencv/opencv/ && make -j$(nproc) && make install && ldconfig && cd / && rm -rf installopencv/
