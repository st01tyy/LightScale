FROM lmsysorg/sglang:v0.4.10.post2-cu126

RUN pip install transformer_engine[pytorch]

RUN cd /root/ && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex/ && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

RUN pip install datasets

RUN pip install setuptools pybind11

RUN pip install tensorboard

RUN pip install pylatexenc==2.10

RUN pip install math-verify==0.7.0

RUN cd /root/ && \
    git clone -b v2.7.4.post1 https://github.com/Dao-AILab/flash-attention && \
    cd flash-attention/ && \
    MAX_JOBS=2 python3 setup.py install

RUN apt-get update && apt-get install -y jq

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ENV http_proxy=
ENV https_proxy=
ENV HTTP_PROXY=
ENV HTTPS_PROXY=

# 设置工作目录
WORKDIR /root/