import os
import sys
import codecs
import subprocess
from setuptools import setup, find_packages


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

install('requests')

setup(
    name='moseq2-app',
    author='Datta Lab',
    description='Interactive data exploration tools for the MoSeq2 pipeline.',
    version=get_version('moseq2_app/__init__.py'),
    platforms=['mac', 'unix'],
    packages=find_packages(),
    setup_requires=['numpy==1.18.3', 'pandas==1.0.5'],
    install_requires=['jupyter-bokeh==2.0.3', 'jupyter==1.0.0', 'ruamel.yaml==0.16.5', 'seaborn==0.11.0',
                      'bokeh==2.2.1', 'fastparquet==0.4.1', 'pandas==1.0.5', 'joblib==0.15.1', 'scipy==1.3.2',
                      'qgrid==1.3.1', 'ipython==7.14.0', 'ipywidgets==7.5.1', 'numpy==1.18.3', 'scikit-image==0.16.2',
                      'scikit-learn==0.20.3', 'opencv-python==4.1.2.30', 'h5py==2.10.0', 'matplotlib==3.1.2', 'plotly==4.14.3',
                      'moseq2-extract @ git+https://github.com/dattalab/moseq2-extract.git@dev',
                      'moseq2-pca @ git+https://github.com/dattalab/moseq2-pca.git@dev',
                      'moseq2-model @ git+https://github.com/dattalab/moseq2-model.git@dev',
                      'moseq2-viz @ git+https://github.com/dattalab/moseq2-viz.git@dev'
                      ],
    python_requires='>=3.6,<3.8'
)
