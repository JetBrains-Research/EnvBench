FROM --platform=linux/amd64 ubuntu:22.04

# Set environment variables
ENV PYENV_ROOT="/root/.pyenv" \
    PATH="/root/.pyenv/bin:/root/.pyenv/shims:/root/.pyenv/versions/3.12.0/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies and additional tools
RUN adduser --force-badname --system --no-create-home _apt \
    && apt-get update -yqq \
    && apt-get install -yqq \
        python3 \
        python3-pip \
        curl \
        wget \
        tree \
        zip \
        unzip \
        git \
        software-properties-common \
        build-essential \
        zlib1g-dev \
        libssl-dev \
        libffi-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        llvm \
        libxml2-dev \
        libxmlsec1-dev

# Install Pyenv and multiple Python versions
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
    && $PYENV_ROOT/bin/pyenv install 3.13.1 \
    && $PYENV_ROOT/bin/pyenv install 3.12.0 \
    && $PYENV_ROOT/bin/pyenv install 3.11.7 \
    && $PYENV_ROOT/bin/pyenv install 3.10.13 \
    && $PYENV_ROOT/bin/pyenv install 3.9.18 \
    && $PYENV_ROOT/bin/pyenv install 3.8.18 \
    && $PYENV_ROOT/bin/pyenv global 3.13.1 \
    && $PYENV_ROOT/bin/pyenv rehash

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/bin/poetry

# Install uv
RUN pip install --no-cache-dir uv

# Install Pyright and other Python tools
RUN pip install --no-cache-dir pyright \
    && pip install search-and-replace \
    && pip install pipenv

# Install Node.js and jq
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs jq

# Remove lists
RUN rm -rf /var/lib/apt/lists/*

# Create and set working directory
RUN mkdir -p /data/project
WORKDIR /data/project

# Make conda always say yes to prompts
ENV CONDA_ALWAYS_YES=true

# Global pyenv versions:
# python3.13 points to 3.13.1, python3.12 points to 3.12.0, ...
RUN pyenv global 3.13.1 3.12.0 3.11.7 3.10.13 3.9.18 3.8.18

# Conda init bash
RUN conda init bash

# Set default shell to bash
CMD ["/bin/bash"]
