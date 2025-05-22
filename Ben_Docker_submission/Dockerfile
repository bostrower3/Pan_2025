# Use official Python base with PyTorch
FROM python:3.10-slim

COPY main.py ./
COPY detector_model.pth ./
COPY roberta-large-local ./roberta-large-local

# Optional: also copy input/output for testing inside the container
# COPY tira-input ./tira-input
# COPY tira-output ./tira-output

RUN pip install torch transformers

# Offline mode
ENV HF_HUB_OFFLINE=1

CMD ["python", "main.py"]