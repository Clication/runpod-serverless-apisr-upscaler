FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:${PATH} \
    APISR_ROOT=/app/APISR \
    APISR_WEIGHT=/app/APISR/pretrained/4x_APISR_RRDB_GAN_generator.pth \
    APISR_WEIGHT_2X=/app/APISR/pretrained/2x_APISR_RRDB_GAN_generator.pth

# ImageMagick + libheif/libaom 讓 AVIF 編碼可用
RUN apt-get update && apt-get install -y --no-install-recommends \
        imagemagick \
        libheif1 \
        libheif-plugin-aomenc \
        libheif-plugin-aomdec \
        libaom3 \
        python3 \
        python3-venv \
        ca-certificates \
        curl \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

# build-time 驗證 AVIF 是否真的可用（不支援就直接讓 build 失敗）
RUN convert -list format | grep -qi 'AVIF' \
    || { echo ">>> 這個映像的 ImageMagick 沒有 AVIF 支援，請改用其他基底映像或自行編譯 IM7"; exit 1; }

WORKDIR /app

# APISR 原始碼
RUN git clone --depth 1 https://github.com/Kiteretsu77/APISR.git "${APISR_ROOT}"

# 4x and 2x RRDB weights are bundled to avoid downloads at worker startup.
RUN mkdir -p "${APISR_ROOT}/pretrained" \
    && wget -q -O "${APISR_WEIGHT}" \
       https://github.com/Kiteretsu77/APISR/releases/download/v0.2.0/4x_APISR_RRDB_GAN_generator.pth \
    && wget -q -O "${APISR_WEIGHT_2X}" \
       https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/2x_APISR_RRDB_GAN_generator.pth \
    && ls -lh "${APISR_WEIGHT}" "${APISR_WEIGHT_2X}"

RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip \
    && python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132 \
    && python -m pip install \
        "numpy<2" \
        opencv-python-headless==4.10.0.84 \
        pillow \
        requests \
        runpod \
        && python -m pip check \
        && python -c "import torch; assert torch.version.cuda == '13.2', torch.version.cuda"

COPY handler.py /app/handler.py

RUN python /app/handler.py --smoke-test

CMD ["python", "-u", "/app/handler.py"]
