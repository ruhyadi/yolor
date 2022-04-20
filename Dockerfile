# syntax=docker/dockerfile:1

# base installation
FROM nvidia/cuda:10.2-base
FROM python:3.8.12-buster

# Install linux packages
RUN apt update && apt install -y --no-install-recommends \
    zip htop screen libgl1-mesa-glx libglib2.0-0 libsm6 \
    libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt

# clone repository
# TODO: change develop branch when fixed
RUN git clone -b develop https://github.com/ruhyadi/yolor /home/yolor

WORKDIR /home/yolor