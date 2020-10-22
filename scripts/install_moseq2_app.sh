#!/bin/bash

# Download some dependencies prior to installing moseq2-app
pip install --upgrade jupyter jupyter-bokeh requests

# Enable qgrid widget extensions
conda install qgrid -y

# Secure credential storage to limit password inputs
git config --global credential.helper store

pip install -e .

# Ensure ffmpeg is installed
conda install -c conda-forge ffmpeg=4.2.0 -y

# Install and Enable widget extensions configurator
conda install -c conda-forge jupyter_nbextensions_configurator -y
jupyter nbextension install --py jupyter_nbextensions_configurator --sys-prefix
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix  --py qgrid

# Install bokeh extensions
jupyter nbextension install --sys-prefix --symlink --py jupyter_bokeh
jupyter nbextension enable jupyter_bokeh --py --sys-prefix
