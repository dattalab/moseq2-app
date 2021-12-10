#!/bin/bash
mkdir test_data
cd test_data

# Config File
curl -L -o config.yaml https://www.dropbox.com/s/nnrzu87z6ch27kx/config_colab.yaml?dl=1

# Progress File
curl -L -o progress.yaml https://www.dropbox.com/s/rwkvt8bc87rgc9r/progress_colab.yaml?dl=1

# Index File
curl -L -o moseq2-index.yaml https://www.dropbox.com/s/ad4zr5a1myo5b8g/moseq2-index_colab.yaml?dl=1

# Aggregate Sessions
curl -L -o aggregate_results.zip https://www.dropbox.com/sh/bfr1o39l7iasgnw/AADRzo-aWAZAHQQP1yzELX1Sa?dl=1 && unzip aggregate_results.zip -d aggregate_results
rm aggregate_results.zip