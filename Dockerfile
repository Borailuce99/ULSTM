FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    python3-pip \
    sudo \
    cmake \
    libjpeg-dev \
    zlib1g-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/* 

RUN python3 -m pip install --upgrade pip

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

RUN pip3 install \
    pandas \
    numpy \
    opencv-python \
    pillow \
    requests \
    matplotlib \
    tqdm==4.66.5 \
    torchsummary \
    torchmetrics \
    lpips

ARG USER_ID
ARG USER_NAME
ARG GROUP_ID
ARG GROUP_NAME

RUN groupadd -g $GROUP_ID $GROUP_NAME \
&& useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash $USER_NAME \
&& echo "$USER_NAME:$USER_NAME" | chpasswd \
&& adduser $USER_NAME sudo

USER $USER_ID

WORKDIR /home/$USER_NAME

CMD ["tail", "-f", "/dev/null"]