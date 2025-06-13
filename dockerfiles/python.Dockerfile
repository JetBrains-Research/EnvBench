FROM ubuntu:22.04

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

# CSV with version and release of python-build-standalone: https://github.com/astral-sh/python-build-standalone
# FIXME: Consider bumping the patch versions of these one day to support `install_only_stripped`
COPY <<EOF /tmp/python-versions.csv
3.13.1,20241205
3.12.0,20231002
3.11.7,20240107
3.10.13,20240224
3.9.18,20240224
3.8.18,20240224
EOF

# Install Pyenv and multiple Python versions
RUN set -eu; \
    git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT; \
    ARCH=$(uname -m); \
    DOWNLOADS="https://github.com/astral-sh/python-build-standalone/releases/download"; \
    while IFS="," read -r VERSION RELEASE; do \
        mkdir -p "$PYENV_ROOT/versions/$VERSION"; \
        wget --quiet "$DOWNLOADS/$RELEASE/cpython-$VERSION+$RELEASE-$ARCH-unknown-linux-gnu-install_only.tar.gz" \
            -O "/tmp/$VERSION.tar.gz"; \
        tar -xzf "/tmp/$VERSION.tar.gz" \
            -C "$PYENV_ROOT/versions/$VERSION" \
            --strip-components=1; \
        rm "/tmp/$VERSION.tar.gz"; \
    done < /tmp/python-versions.csv; \
    pyenv rehash

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Linux-$(uname -m).sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

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
# python3.13 points to 3.13.5, python3.12 points to 3.12.11, ...
RUN pyenv global $(cut -d"," -f1 /tmp/python-versions.csv | tr "\n" " ") \
    && rm /tmp/python-versions.csv

# Conda init bash
RUN conda init bash

# Set default shell to bash
CMD ["/bin/bash"]
