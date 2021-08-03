#!/bin/bash
mkdir test_data
cd test_data

# amphetamine data
curl -L -o amphetamine_example_0.zip https://www.dropbox.com/sh/gnx4htuvmb7ue65/AACB-twQSGiERqkHHEUiLc5ya?dl=1 && unzip amphetamine_example_0.zip -d amphetamine_example_0
rm amphetamine_example_0.zip

# saline data
curl -L -o saline_example_0.zip https://www.dropbox.com/sh/ca9nbj2vlsn8hyq/AACuBEODDwaizuITN6DjXv39a?dl=1 && unzip saline_example_0.zip -d saline_example_0
rm saline_example_0.zip