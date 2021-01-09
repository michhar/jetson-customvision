########################################################################
#
# Dockerfile to build ARM64 (aarch64) image with CUDA and LVA Custom 
# Vision Service tutorial.
#
#######################################################################
FROM nvcr.io/nvidia/l4t-tensorflow:r32.4.4-tf2.3-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev unzip \
    make cmake automake gcc g++ pkg-config \
    python3-numpy python3-opencv python3-h5py \
    libhdf5-serial-dev hdf5-tools libhdf5-dev \
    libhdf5-100 zlib1g-dev zip libjpeg8-dev liblapack-dev \
    libblas-dev gfortran vim \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean \
    && apt purge -y --auto-remove wget

RUN pip3 install flask~=1.1.2 pillow~=7.2.0

COPY app /app

# Expose the port
EXPOSE 80

# Set the working directory
WORKDIR /app

# Set the CUDA library and bin folder
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
RUN echo "export PATH=/usr/local/cuda-10.2/bin:${PATH:+:${PATH}}" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

# Run the flask server for the endpoints
CMD python3 -u app.py
