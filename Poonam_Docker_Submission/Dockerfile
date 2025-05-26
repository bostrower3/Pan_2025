FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04



COPY requirements.txt .
# ✅ Install Python + pip
RUN apt-get update && \
    apt-get install -y python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install -r requirements.txt --no-cache --break-system-packages

RUN python3 -c 'import nltk; nltk.download("punkt"); nltk.download("punkt_tab");'

# Environment variables for offline mode
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy all necessary files and folders
COPY xgb_model.pkl .
COPY rf_model.pkl .
COPY tfidf_vectorizer.pkl .
COPY main.py .

# Run the main script
CMD ["python", "main.py"]
