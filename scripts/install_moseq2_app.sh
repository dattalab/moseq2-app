#!/bin/bash

# Ensure latest version of jupyter is installed
pip install --upgrade jupyter jupyter_bokeh

pip install -e .

# Ensure ffmpeg is installed
conda install -c conda-forge ffmpeg==4.2.0 -y

# Install and Enable widget extensions configurator
jupyter nbextension install --py jupyter_nbextensions_configurator --sys-prefix
jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Enable qgrid widget extensions
conda install qgrid -y
jupyter nbextension enable --py --sys-prefix  --py qgrid

# Install bokeh extensions
jupyter nbextension install --sys-prefix --symlink --py jupyter_bokeh
jupyter nbextension enable jupyter_bokeh --py --sys-prefix

# Enable jupyter table of contents
jupyter nbextension enable toc2/main