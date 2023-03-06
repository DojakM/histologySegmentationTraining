FROM mlfcore/base:1.0.0

# Install the conda environment
RUN sudo apt-get update
RUN sudo DEBIAN_FRONTEND="noninteractive"  apt-get -y install tzdata
RUN sudo apt-get install -y --reinstall openmpi-bin libopenmpi-dev
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -a

# Activate the environment
RUN echo "source activate seg_training" >> ~/.bashrc
ENV PATH /home/user/miniconda/envs/seg_training/bin:$PATH

# Dump the details of the installed packages to a file for posterity
RUN conda env export --name seg_training > seg_training_environment.yml

# Currently required, since mlflow writes every file as root!
USER root
