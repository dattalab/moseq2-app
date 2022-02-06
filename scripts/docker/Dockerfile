# FROM continuumio/miniconda3 AS condabuild
FROM condaforge/mambaforge:4.11.0-0 AS condabuild

WORKDIR /root

COPY moseq2-app /root/moseq2-app
COPY moseq2-extract /root/moseq2-extract
COPY moseq2-pca /root/moseq2-pca
COPY moseq2-viz /root/moseq2-viz
COPY moseq2-model /root/moseq2-model

# add a default password to notebook: moseq
RUN mkdir -p /root/.jupyter
COPY jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json
COPY copy-notebooks-download-scripts.py .

# only need g++ to install moseq2-model
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/opt/conda/pkgs apt update && \
    apt install -y g++ libglib2.0-0 libsm6 libxrender-dev libxext6 curl unzip && \
    mamba env update -n base --file /root/moseq2-app/scripts/moseq2-env.yaml && \
    jupyter nbextensions_configurator enable --sys-prefix && \
    pip install ./moseq2-extract && \
    pip install ./moseq2-pca && \
    pip install ./moseq2-model && \
    pip install ./moseq2-viz && \
    cd moseq2-app && \
    ./scripts/install_moseq2_app.sh && \
    cd .. && \
    pip cache purge && \
    rm -rf moseq2-extract moseq2-pca moseq2-model moseq2-viz && \
    conda clean -ay && \
    apt purge -y g++ && \
    apt clean -y && \
    apt autoremove -y && \
    cp /root/moseq2-app/notebooks/*.ipynb . && \
    cp /root/moseq2-app/scripts/*.sh . && \
    chmod +x ./copy-notebooks-download-scripts.py

VOLUME [ "/data" ]
EXPOSE 8888
EXPOSE 8787

ENTRYPOINT ./copy-notebooks-download-scripts.py && SHELL=/bin/bash jupyter notebook --notebook-dir=/data --port=8888 --ip=0.0.0.0 --no-browser --allow-root