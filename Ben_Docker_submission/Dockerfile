FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN apt-get update \
  && apt-get install -y python3-pip \
  && pip3 install torch transformers

# Offline mode
ENV HF_HUB_OFFLINE=1

COPY main.py ./
COPY detector_model.pth ./
COPY roberta-large-local ./roberta-large-local

CMD ["python", "main.py"]
