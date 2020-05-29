# ----------------------------------------------- Base Docker Image ----------------------------------------------- 
FROM nvidia/cuda:10.0-devel-ubuntu18.04

ENV PYTORCH_VERSION=1.1.0
ENV TORCHVISION_VERSION=0.3.0
ENV CUDNN_VERSION=7.6.0.64-1+cuda10.0
ENV NCCL_VERSION=2.4.7-1+cuda10.0

# Workaround for deadlock issue. To be removed with next major Pytorch release
ENV NCCL_LL_THRESHOLD=0

# Python 2.7 or 3.6 is supported by Ubuntu Bionic out of the box
ARG python=3.6
ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-4.8 \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    libcudnn7=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION} \
    libnccl-dev=${NCCL_VERSION} \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    unzip \
    bzip2 \
    htop

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Instal Python and pip
RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
    apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install PyTorch
RUN pip install future typing numpy awscli
RUN pip install https://download.pytorch.org/whl/cu100/torch-${PYTORCH_VERSION}-cp36-cp36m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cu100/torchvision-${TORCHVISION_VERSION}-cp36-cp36m-linux_x86_64.whl


# Configure environment variables - default working directory is "/workspace"
WORKDIR /workspace
ENV PYTHONPATH="/workspace"

ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}

ARG AWS_DEFAULT_REGION
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

ARG WANDB_ENTITY
ENV WANDB_ENTITY=${WANDB_ENTITY}

ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Settings for S3
RUN aws configure set default.s3.max_concurrent_requests 100 && \
    aws configure set default.s3.max_queue_size 10000

# ----------------------------------------------- Repo specific dependencies ----------------------------------------------- 

# Install dependencies
RUN pip install ninja yacs cython matplotlib opencv-python tqdm wandb mpi4py onnx==1.5.0 onnxruntime coloredlogs scipy pycuda
RUN pip uninstall -y pillow
RUN pip install pillow-simd
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotools'

WORKDIR /workspace
ENV PYTHONPATH="/workspace/ouroboros:$PYTHONPATH"
RUN echo "10.100.0.142    github.awsinternal.tri.global" >> /etc/hosts \
  && git clone https://github.awsinternal.tri.global/quincy-chen/ouroboros.git \
  && cd ouroboros \
  && pip install -r requirements.txt \
  && cd ouroboros/dgp \
  && git checkout key-point-ann-obj \
  && git checkout 9f0fa10c8fe536bdc8f0ace57cce288ad08c2f11 \
  && pip install -r requirements.txt \
  && make build-proto

# Install detr
ENV PYTHONPATH="/workspace/detr:$PYTHONPATH"
COPY . /workspace/detr
WORKDIR /workspace/detr
# Fix pycocotools python3 compatibility issue.
RUN sed -i "s|unicode|bytes|g" /usr/local/lib/python3.6/dist-packages/pycocotools/coco.py

