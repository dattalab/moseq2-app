#!/bin/bash

# Download some dependencies prior to installing moseq2-app
pip install --upgrade jupyter jupyter-bokeh requests opencv-python==4.1.2.30 scipy==1.3.2

# Secure credential storage to limit password inputs
#git config --global credential.helper store

pip install -e .

# Install and Enable widget extensions configurator
jupyter nbextension install --py jupyter_nbextensions_configurator --sys-prefix
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix  --py qgrid

# Install bokeh extensions
jupyter nbextension install --sys-prefix --symlink --py jupyter_bokeh
jupyter nbextension enable jupyter_bokeh --py --sys-prefix
