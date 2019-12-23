#!/bin/bash
conda env create -f moseq2-env.yaml
conda activate moseq2dev
export CC=/usr/local/bin/gcc-7
export CXX=/usr/local/bin/g++-7
pip install git+https://github.com/dattalab/moseq2-model.git@dev