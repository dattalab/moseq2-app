#!/bin/bash
conda env create -f moseq2-env.yaml
conda init
conda activate moseq2
read -p "Would you like to try auto-installing GCC-7? [y/n]" y_gcc
stry="y"
if [ "$y_gcc" == "$stry" ]; then
  {
    ./install_gcc.sh
  }
fi
pip install git+https://github.com/dattalab/moseq2-model.git@release
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user