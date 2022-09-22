FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV TZ=Asiz/Seoul
ENV TERM=xterm-256color

RUN ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime

#### 0. Install python and pip
RUN apt-get -y update && apt-get install -y git wget curl
RUN apt-get update
RUN apt-get upgrade python3 -y
RUN apt-get install python3-pip -y
RUN alias python='python3'

#### 1. Install Pytorch
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

#### 2. Install other dependencies
WORKDIR /usr/app
COPY . ./
RUN pip install -r ./requirements.txt

#### 3. Clone external codes
RUN git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
RUN git clone https://github.com/LeviBorodenko/motionblur motionblur

#### 4. change user
RUN useradd docker_user -u 1000 -m
