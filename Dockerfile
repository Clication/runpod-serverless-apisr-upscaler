FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:${PATH} \
    APISR_ROOT=/app/APISR \
    APISR_WEIGHT=/app/APISR/pretrained/4x_APISR_RRDB_GAN_generator.pth \
    APISR_WEIGHT_2X=/app/APISR/pretrained/2x_APISR_RRDB_GAN_generator.pth \
    REAL_CUGAN_ROOT=/app/Real-CUGAN \
    MANGA_TRANSLATOR_ROOT=/app/manga-image-translator \
    XDG_CACHE_HOME=/opt/model-cache \
    XDG_CONFIG_HOME=/opt/model-config

# ImageMagick + libheif/libaom 讓 AVIF 編碼可用
RUN apt-get update && apt-get install -y --no-install-recommends \
        imagemagick \
        libheif1 \
        libheif-plugin-aomenc \
        libheif-plugin-aomdec \
        libaom3 \
        libgl1 \
        libglib2.0-0t64 \
        python3 \
        python3-venv \
        ca-certificates \
        curl \
        git \
        unzip \
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

# Bundle the official Real-CUGAN implementation and 2x/4x weights.
RUN mkdir -p "${REAL_CUGAN_ROOT}" \
    && upstream=https://raw.githubusercontent.com/bilibili/ailab/2799af78ef105b414cc4b796c67c8511acdcdf6f/Real-CUGAN \
    && curl -fsSL "${upstream}/upcunet_v3.py" -o "${REAL_CUGAN_ROOT}/upcunet_v3.py" \
    && curl -fsSL "${upstream}/LICENSE" -o "${REAL_CUGAN_ROOT}/LICENSE" \
    && curl -fsSL https://github.com/bilibili/ailab/releases/download/Real-CUGAN/updated_weights.zip \
       -o /tmp/real-cugan-weights.zip \
    && echo '99a8ce2e4a6ce0548668879f148b3db3e143201d3df42c825a4143490396da53  /tmp/real-cugan-weights.zip' | sha256sum -c - \
    && unzip -j /tmp/real-cugan-weights.zip \
       'updated_weights/up2x-latest-no-denoise.pth' 'updated_weights/up4x-latest-no-denoise.pth' \
       'updated_weights/up2x-latest-denoise[123]x.pth' 'updated_weights/up4x-latest-denoise3x.pth' \
       -d "${REAL_CUGAN_ROOT}" \
    && rm /tmp/real-cugan-weights.zip

RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip \
    && python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132 \
    && python -m pip install \
        "numpy<2" \
        opencv-python==4.10.0.84 \
        pillow \
        pcleaner-cli==2.11.11 \
        simple-lama-inpainting==0.1.1 \
        einops==0.8.2 \
        omegaconf==2.3.0 \
        py3langid==0.2.2 \
        requests \
        runpod \
        && python -m pip check \
        && python -c "import torch; assert torch.version.cuda == '13.2', torch.version.cuda"

# Fetch both text detectors plus the OCR and inpainting models at build time.
# The CLI may report a download failure without a nonzero exit code.
RUN pcleaner-cli load models --both \
    && test -s "${XDG_CACHE_HOME}/pcleaner/model/comictextdetector.pt" \
    && test -s "${XDG_CACHE_HOME}/pcleaner/model/comictextdetector.pt.onnx" \
    && test -s "${XDG_CACHE_HOME}/pcleaner/model/anime-manga-big-lama.pt"

ENV HF_HUB_OFFLINE=1

# Bundle only Manga Image Translator's detection and inpainting modules. The
# upstream entry points import the translation, OCR, and rendering stacks.
RUN mkdir -p "${MANGA_TRANSLATOR_ROOT}" \
    && git init /tmp/manga-translator-src \
    && git -C /tmp/manga-translator-src fetch --depth 1 https://github.com/zyddnys/manga-image-translator.git 6851ca7e207f76deb2b2317d9e513bc05a4bf8d3 \
    && git -C /tmp/manga-translator-src archive --output=/tmp/manga-translator.tar FETCH_HEAD \
        LICENSE manga_translator/config.py manga_translator/utils \
        manga_translator/detection/common.py manga_translator/detection/ctd.py manga_translator/detection/ctd_utils \
        manga_translator/inpainting/common.py manga_translator/inpainting/inpainting_lama_mpe.py \
    && tar -xf /tmp/manga-translator.tar -C "${MANGA_TRANSLATOR_ROOT}" \
    && touch "${MANGA_TRANSLATOR_ROOT}/manga_translator/__init__.py" \
        "${MANGA_TRANSLATOR_ROOT}/manga_translator/detection/__init__.py" \
        "${MANGA_TRANSLATOR_ROOT}/manga_translator/inpainting/__init__.py" \
    && rm -rf /tmp/manga-translator-src /tmp/manga-translator.tar \
    && mkdir -p "${MANGA_TRANSLATOR_ROOT}/models/detection" "${MANGA_TRANSLATOR_ROOT}/models/inpainting" \
    && ln -s "${XDG_CACHE_HOME}/pcleaner/model/comictextdetector.pt" "${MANGA_TRANSLATOR_ROOT}/models/detection/comictextdetector.pt" \
    && ln -s "${XDG_CACHE_HOME}/pcleaner/model/comictextdetector.pt.onnx" "${MANGA_TRANSLATOR_ROOT}/models/detection/comictextdetector.pt.onnx" \
    && cd "${MANGA_TRANSLATOR_ROOT}/models/inpainting" \
    && curl -fsSL --retry 3 https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt -o lama_large_512px.ckpt \
    && echo '11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935  lama_large_512px.ckpt' | sha256sum -c -

COPY handler.py manga_cleaner.py /app/

RUN python /app/handler.py --smoke-test

CMD ["python", "-u", "/app/handler.py"]
