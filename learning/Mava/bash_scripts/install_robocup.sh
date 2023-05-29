export DEBIAN_FRONTEND=noninteractive
sudo apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python-opengl

VERSION=16.0.1

# fetch repo / ppa packages, etc
sudo apt-get -y update --fix-missing

# Install package, that hangs the operation, separately
sudo apt install -y tzdata

sudo apt update && \
    apt -y install autoconf bison clang flex libboost-dev libboost-all-dev libc6-dev make wget

sudo apt -y install build-essential libboost-all-dev qt5-default libfontconfig1-dev libaudio-dev libxt-dev libglib2.0-dev libxi-dev libxrender-dev libboost-all-dev

sudo apt-get -y install vim

sudo wget https://github.com/rcsoccersim/rcssserver/archive/rcssserver-$VERSION.tar.gz && \
    tar xfz rcssserver-$VERSION.tar.gz && \
    cd rcssserver-rcssserver-$VERSION && \
    # Temp fix for ubuntu 20.04 - https://github.com/rcsoccersim/rcssserver/issues/50
    vim -c "%s/-lrcssclangparser/librcssclangparser.la/g|wq" src/Makefile.am && \
    ./bootstrap && \
    ./configure && \
    make && \
    make install && \
    ldconfig

sudo wget https://github.com/rcsoccersim/rcssmonitor/archive/rcssmonitor-$VERSION.tar.gz && \
    tar xfz rcssmonitor-$VERSION.tar.gz && \
    cd rcssmonitor-rcssmonitor-$VERSION && \
    ./bootstrap && \
    ./configure && \
    make && \
    make install && \
    ldconfig

sudo ldconfig && \
    apt update && \
    apt install -y libboost-filesystem1.71.0 libboost-system1.71.0 libboost-program-options-dev tmux

sudo apt-get install -y libqt5widgets5
