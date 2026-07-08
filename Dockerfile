FROM rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.11_pytorch_release_2.5.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/model_cache

WORKDIR /app

COPY requirement.txt .

RUN pip install --upgrade pip && \
    pip install \
        transformers>=4.49.0 \
        accelerate>=0.30.0 \
        qwen-vl-utils>=0.0.8 \
        httpx>=0.27.0 \
        rich>=13.0.0 \
        opencv-python-headless>=4.9.0 \
        Pillow>=10.0.0 \
        pydantic>=2.0.0 && \
    rm -rf /root/.cache/pip

COPY app/ ./app/

RUN mkdir -p /app/input /app/output /model_cache

VOLUME ["/app/input", "/app/output", "/model_cache"]

CMD ["python3", "app/main.py"]
