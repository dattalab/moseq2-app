#!/bin/bash
mkdir test_data
cd test_data

# Config file
curl -L -o config.yaml https://www.dropbox.com/s/ahjpgrgmf24zu8i/config_colab_withresults.yaml?dl=0

# Progress File
curl -L -o progress.yaml https://www.dropbox.com/s/qn6fycq7erw6www/progress_colab_withresults.yaml?dl=0

# Index File
curl -L -o moseq2-index.yaml  https://www.dropbox.com/s/xeqsu2q7lhwbvii/moseq2-index_colab_withresults.yaml?dl=0

# PCA Files
curl -L -o _pca.zip https://www.dropbox.com/sh/fhhietihmxud9tb/AACYUbCiZmtxkqb-QDoymfnCa?dl=0 && unzip _pca.zip -d _pca
rm _pca.zip

# Modeling Files
curl -L -o saline-amphetamine.zip https://www.dropbox.com/sh/dpaiacewonlh7nb/AABlal7f2TCyBtMTUbjhHoBDa?dl=0  && unzip saline-amphetamine.zip -d saline-amphetamine
rm saline-amphetamine.zip
