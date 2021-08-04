#!/bin/bash
mkdir test_data
cd test_data

# Config File
curl -L -o config.yaml https://www.dropbox.com/s/2zayp2gbz5e4ssf/config_colab.yaml?dl=0

# Progress File
curl -L -o progress.yaml https://www.dropbox.com/s/i51yfglirupi32h/progress_colab.yaml?dl=0

# Index File
curl -L -o moseq2-index.yaml https://www.dropbox.com/s/5oq5aq8jly7laic/moseq2-index_colab.yaml?dl=0

# Aggregate Sessions
curl -L -o aggregate_results.zip https://www.dropbox.com/sh/fdi8mq4a2s4xp9q/AABlU230uGutBI6oA7QhrMkOa?dl=1 && unzip aggregate_results.zip -d aggregate_results
rm aggregate_results.zip