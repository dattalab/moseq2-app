#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  conda install -c anaconda clang_osx-64 -y
  conda install -c anaconda clangxx_osx-64 -y
else
  conda install -c anaconda gcc_linux-64 -y
  conda install -c anaconda gxx_linux-64 -y
fi

pip install -e .

# Install and Enable widget extensions configurator
jupyter nbextension install --py jupyter_nbextensions_configurator --sys-prefix
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix  --py qgrid

# Install bokeh extensions
jupyter nbextension install --sys-prefix --symlink --py jupyter_bokeh
jupyter nbextension enable jupyter_bokeh --py --sys-prefix
