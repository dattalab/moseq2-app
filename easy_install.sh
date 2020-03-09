#!/bin/bash
conda env create -f moseq2-env.yaml
conda activate moseq2dev
./install_gcc.sh
pip install git+https://github.com/dattalab/moseq2-model.git@release
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user