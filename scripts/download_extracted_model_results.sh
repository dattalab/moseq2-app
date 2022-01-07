#!/bin/bash
mkdir test_data
cd test_data


# Config file
curl -L -o config.yaml https://www.dropbox.com/s/5nb3rih0kr9d20b/config_modeled.yaml?dl=0

# Progress File
curl -L -o progress.yaml https://www.dropbox.com/s/pu77qvk77uvn7s4/progress_modeled.yaml?dl=0

# Index File
curl -L -o moseq2-index.yaml  https://www.dropbox.com/s/hantgvvai1r0xbo/moseq2-index_modeled.yaml?dl=0

# Aggregate Sessions
curl -L -o aggregate_results.zip https://www.dropbox.com/sh/bfr1o39l7iasgnw/AADRzo-aWAZAHQQP1yzELX1Sa?dl=1 && unzip aggregate_results.zip -d aggregate_results
rm aggregate_results.zip

# PCA Files
curl -L -o _pca.zip https://www.dropbox.com/sh/fhhietihmxud9tb/AACYUbCiZmtxkqb-QDoymfnCa?dl=0 && unzip _pca.zip -d _pca
rm _pca.zip

# Modeling Files
curl -L -o saline-amphetamine.zip https://www.dropbox.com/sh/dpaiacewonlh7nb/AABlal7f2TCyBtMTUbjhHoBDa?dl=0  && unzip saline-amphetamine.zip -d saline-amphetamine
rm saline-amphetamine.zip

