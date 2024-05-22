FROM  nvidia/cuda:12.2.2-devel-ubuntu20.04
ENV PATH /opt/conda/bin:$PATH
WORKDIR /opt/app

RUN apt-get update --fix-missing && \
    apt-get install -y wget git&& \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh 
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda 

RUN echo "source activate base" > ~/.bashrc
RUN conda install -y python=3.9
RUN conda install pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
RUN pip install transformers==4.38.0 accelerate==0.27.2 datasets==2.17.1 deepspeed==0.13.2 sentencepiece wandb
RUN pip install flash-attn==2.3.4 --no-build-isolation
CMD ["bash"]