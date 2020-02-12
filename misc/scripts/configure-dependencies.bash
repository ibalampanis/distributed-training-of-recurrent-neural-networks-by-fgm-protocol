#!/bin/bash
# Run this script as superuser

# Keep Ubuntu up to date
sudo apt-get -y update
sudo apt-get -y upgrade      # Comment in order not to install new versions of packages currently installed
sudo apt-get -y dist-upgrade # Comment in order not to handle changing dependencies with new vers. of pack.
sudo apt-get -y autoremove   # Comment in order not to remove packages that are now no longer needed

# OS libraries
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libx11-dev libatlas-base-dev
sudo apt-get install -y libgtk-3-dev libboost-python-dev

# # omnet++
sudo apt-get install build-essential gcc g++ bison flex perl \python python3 qt5-default libqt5opengl5-dev tcl-dev tk-dev \libxml2-dev zlib1g-dev default-jre doxygen graphviz libwebkitgtk-1.0 openscenegraph-plugin-osgearth libosgearth-dev openmpi-bin libopenmpi-dev libpcap-dev gnome-color-chooser nemiver libjsoncpp-dev
wget https://github.com/omnetpp/omnetpp/releases/download/omnetpp-5.6/omnetpp-5.6-src-linux.tgz
tar xvfz omnetpp-5.6-src-linux.tgz
cd omnetpp-5.6
./configure
make -j4
export PATH=$PATH:/home/ibalampanis/omnetpp-5.6/bin
cd

# dsarch
sudo apt-get install -y cxxtest libboost-all-dev doxygen autoconf-archive dvips-fontdata-n2bk texlive-latex-recommended gawk graphviz

# hdf5
sudo apt-get install -y libhdf5-dev

# armadillo
sudo apt-get install -y cmake libopenblas-dev liblapack-dev libarpack2-dev
wget http://sourceforge.net/projects/arma/files/armadillo-9.800.3.tar.xz
tar xvf armadillo-9.800.3.tar.xz
cd armadillo-9.800.3/
./configure
make -j4
sudo make install
cd

# # CUDA 9
sudo apt-get install -y g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
sudo apt-get install -y gcc-6 g++-6
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
chmod +x cuda_9.0.176_384.81_linux.run
sudo ./cuda_9.0.176_384.81_linux.run --override

# CuDNN 7
# In home folder must be the archived file 'cudnn-9.0-linux-x64-v7.6.4.38.tgz'
tar -xzvf cudnn-9.0-linux-x64-v7.6.4.38.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-9.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*

# Python libraries
sudo apt-get install -y python-dev python-pip python3-dev python3-pip

# dlib
wget http://dlib.net/files/dlib-19.10.tar.bz2
tar xvf dlib-19.10.tar.bz2
cd dlib-19.10/
mkdir build
cd build
cmake ..
cmake --build . --config Release
sudo make install
sudo ldconfig
cd

# blas and lapack
sudo apt-get install -y libblas-dev liblapack-dev
sudo apt-get install -y libblas-dev checkinstall
sudo apt-get install -y liblapacke-dev checkinstall
sudo apt-get install -y liblapack-doc checkinstall

# mlpack
sudo apt-get install -y libmlpack-dev
wget https://www.mlpack.org/files/mlpack-3.2.2.tar.gz
tar -xvzpf mlpack-3.2.2.tar.gz
mkdir mlpack-3.2.2/build && cd mlpack-3.2.2/build
cmake ../
make -j4  # The -j is the number of cores you want to use for a build.
sudo make install
export LD_LIBRARY_PATH=/usr/local/lib
cd

# open-cv
sudo apt-get install -y build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev
sudo apt-get install -y libopencv-dev
