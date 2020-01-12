FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libsm6 libxext6 libxrender-dev\
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=3.6 numpy pyyaml scipy ipython mkl mkl-include ninja cython opencv typing pillow && \
     /opt/conda/bin/conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch && \ 
     /opt/conda/bin/conda clean -ya

ADD setup.py .
ADD requirements.txt .

COPY xv xv/
COPY test test/

RUN /opt/conda/bin/pip install -r requirements.txt

# Download weights for models
RUN /opt/conda/bin/python -c "import segmentation_models_pytorch as smp; smp.encoders.get_encoder('efficientnet-b1', 'imagenet')"
RUN /opt/conda/bin/python -c "import segmentation_models_pytorch as smp; smp.encoders.get_encoder('efficientnet-b2', 'imagenet')"
RUN /opt/conda/bin/python -c "import segmentation_models_pytorch as smp; smp.encoders.get_encoder('efficientnet-b3', 'imagenet')"
RUN /opt/conda/bin/python -c "import segmentation_models_pytorch as smp; smp.encoders.get_encoder('efficientnet-b4', 'imagenet')"
RUN /opt/conda/bin/python -c "import segmentation_models_pytorch as smp; smp.encoders.get_encoder('efficientnet-b5', 'imagenet')"
RUN /opt/conda/bin/python -c "import segmentation_models_pytorch as smp; smp.encoders.get_encoder('efficientnet-b6', 'imagenet')"
RUN /opt/conda/bin/python -c "import segmentation_models_pytorch as smp; smp.encoders.get_encoder('efficientnet-b7', 'imagenet')"

ENTRYPOINT ["/opt/conda/bin/python", "xv/"]
