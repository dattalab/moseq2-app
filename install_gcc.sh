#!/bin/bash
OS="$(uname)"
DARWIN="Darwin"
LINUX="Linux"
CYGWIN="Cygwin"

if [ "$OS" == "$DARWIN" ]; then
  {
    xcode-select --install ||
    brew install gcc@7 ||
    CC_PATH="$(which gcc-7)"
    CXX_PATH="$(which g++-7)"
    export CC=$CC_PATH
    export CXX=$CXX_PATH
  } ||
  {
    CC_PATH="$(which gcc-7)"
    CXX_PATH="$(which g++-7)"
    export CC=$CC_PATH
    export CXX=$CXX_PATH
  }
fi

if [ $OS == "$LINUX" ]; then
  { # try ubuntu install
    sudo apt install software-properties-common ||
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test ||
    sudo apt install gcc-7 g++-7 ||
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 --slave /usr/bin/g++ g++ /usr/bin/g++-7 --slave /usr/bin/gcov gcov /usr/bin/gcov-7 ||
    CC_PATH="$(which gcc-7)"
    CXX_PATH="$(which g++-7)"
    export CC=$CC_PATH
    export CXX=$CXX_PATH
  } ||
  {
    yum install centos-release-scl ||
    yum install devtoolset-7-gcc-c++ ||
    CC_PATH="$(which gcc-7)"
    CXX_PATH="$(which g++-7)"
    export CC=$CC_PATH
    export CXX=$CXX_PATH
  } ||
  {
    CC_PATH="$(which gcc-7)"
    CXX_PATH="$(which g++-7)"
    export CC=$CC_PATH
    export CXX=$CXX_PATH
  }
fi

if [ $OS == "$CYGWIN" ]; then
  wget http://mirror.team-cymru.com/gnu/gcc/gcc-7.2.0/gcc-7.2.0.tar.gz &&
  tar xf gcc-7.2.0.tar.gz &&
  mkdir build-gcc &&
  cd build-gcc &&
  ../gcc-7.2.0/configure --program-suffix=-7.2.0 --enable-languages=c,c++ --disable-bootstrap --disable-shared ||
  CC_PATH="$(which gcc-7)"
  CXX_PATH="$(which g++-7)"
  export CC=$CC_PATH
  export CXX=$CXX_PATH
fi