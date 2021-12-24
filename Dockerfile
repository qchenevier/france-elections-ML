ARG BASE_IMAGE=condaforge/mambaforge
FROM $BASE_IMAGE

# install project requirements
COPY src/.condarc /root/.condarc
COPY src/pip.conf /root/pip.conf
ENV PIP_CONFIG_FILE /root/pip.conf
COPY src/environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && \
    mamba clean --all --yes
RUN echo "source activate france_elections_ML" > ~/.bashrc
ENV PATH /opt/conda/envs/france_elections_ML/bin:$PATH

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
useradd -d /home/kedro -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro

# copy the whole project except what is in .dockerignore
WORKDIR /home/kedro
COPY . .
RUN chown -R kedro:${KEDRO_GID} /home/kedro
USER kedro
RUN chmod -R a+w /home/kedro

EXPOSE 8888

ENV NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE true
CMD ["kedro", "run"]
