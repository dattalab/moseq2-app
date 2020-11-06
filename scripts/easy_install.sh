#!/bin/bash
option='0'
one='1'
two='2'
echo "Welcome to the MoSeq2-App Installer!"
echo "Enter your desired installation method."
echo "  1. Create a new conda environment named moseq2-app."
echo "  2. Install the latest MoSeq2 tools in your current conda environment."
echo "  Else to exit."
read option
if [ $option == $one ]; then
{
  echo 'Installing conda environment named moseq2-app'
  export CC="$(which gcc-7)"
  export CXX="$(which g++-7)"
} ||
{
  echo "gcc-7 not found. Trying default gcc version..."
  export CC="$(which gcc)"
  export CXX="$(which g++)"
} &&
{
  conda env create -f scripts/moseq2-env.yaml
  conda init
  echo "Conda environment installation succeeded, now activate your environment and install moseq2-app."
  echo "You can run this script again and enter 2 to install all the requirements."
}
fi

if [ $option == $two ]; then
{
  echo 'Installing latest version of moseq2'
  export CC="$(which gcc-7)"
  export CXX="$(which g++-7)"
} ||
{
  echo "gcc-7 not found. Trying default gcc version..."
  export CC="$(which gcc)"
  export CXX="$(which g++)"
} &&
{
  ./scripts/install_moseq2_app.sh
}
fi
