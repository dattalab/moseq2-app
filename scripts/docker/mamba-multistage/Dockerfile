FROM continuumio/miniconda3 AS condabuild

WORKDIR /root

COPY moseq2-app /root/moseq2-app
COPY moseq2-extract /root/moseq2-extract
COPY moseq2-pca /root/moseq2-pca
COPY moseq2-viz /root/moseq2-viz
COPY moseq2-model /root/moseq2-model

# add a default password to notebook: moseq
RUN mkdir -p /root/.jupyter
COPY jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json

RUN cp /root/moseq2-app/notebooks/*.ipynb .

# only need g++ to install moseq2-model
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y g++

RUN conda install -c conda-forge mamba
RUN --mount=type=cache,target=/opt/conda/pkgs mamba env create -n moseq --file /root/moseq2-app/scripts/moseq2-env.yaml
RUN mamba install -c conda-forge conda-pack

RUN echo "conda activate moseq" >> ~/.bashrc

# run this with the multi-stage build
SHELL [ "/bin/bash", "--login", "-c" ]
RUN jupyter nbextensions_configurator enable --sys-prefix

RUN pip install ./moseq2-extract && pip install ./moseq2-pca \
    && pip install ./moseq2-model && pip install ./moseq2-viz \
    && cd moseq2-app && ./scripts/install_moseq2_app.sh
RUN pip cache purge && rm -rf moseq2-extract moseq2-pca moseq2-model moseq2-viz \
    && conda clean -ay && apt purge -y g++ && apt clean -y && apt autoremove -y

# # run this with the multi-stage build
RUN conda activate base; conda-pack -n moseq -o /tmp/env.tar && \
mkdir /venv && cd /venv && tar -xf /tmp/env.tar && \
rm /tmp/env.tar

RUN /venv/bin/conda-unpack

# start a multi-stage build
FROM debian:bullseye-slim AS runtime

COPY --from=condabuild /venv /venv
COPY --from=condabuild /root/*.ipynb .
COPY jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json
COPY --from=condabuild /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt

RUN apt update && apt install -y libglib2.0-0 libsm6 libxrender-dev libxext6 && apt purge -y && apt clean -y && apt autoremove -y

COPY copy-notebooks.py .
RUN chmod +x ./copy-notebooks.py
VOLUME [ "/data" ]
EXPOSE 8888

RUN echo "source /venv/bin/activate" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]

ENTRYPOINT source /venv/bin/activate && \
./copy-notebooks.py && \
SHELL=/bin/bash jupyter notebook --notebook-dir=/data --port=8888 --ip=0.0.0.0 --no-browser --allow-root
