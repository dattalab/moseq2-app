#!/bin/bash
mkdir test_data
cd test_data

# Config file
curl -L -o config.yaml https://www.dropbox.com/s/iy7l5kolzhz2wyv/config_colab_withresults.yaml?dl=0

# Progress File
curl -L -o progress.yaml https://www.dropbox.com/s/lkkrc2xmj1u17zh/progress_colab_withresults.yaml?dl=0

# Index File
curl -L -o moseq2-index.yaml  https://www.dropbox.com/s/8hzw8nhpmwabn16/moseq2-index_colab_withresults.yaml?dl=0

# PCA Files
curl -L -o _pca.zip https://www.dropbox.com/sh/ht36mw2a04h0a1r/AAAjMIFTFva4zr7vs_JavyZva?dl=0 && unzip _pca.zip -d _pca
rm _pca.zip

# Modeling Files
curl -L -o saline-amphetamine.zip https://www.dropbox.com/sh/qnsrbpqrbot0lw2/AAAB3HBUGLVAY1HxWztY6RKla?dl=0  && unzip saline-amphetamine.zip -d saline-amphetamine
rm saline-amphetamine.zip
