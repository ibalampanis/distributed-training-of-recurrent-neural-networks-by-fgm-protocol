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
sudo apt-get install -y libgtk-3-dev libboost-python-dev python-lxml

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

# Python libraries
sudo apt-get install -y python-dev python-pip python3-dev python3-pip

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
