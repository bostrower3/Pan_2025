FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04



COPY requirements.txt .
# ✅ Install Python + pip
RUN apt-get update && \
    apt-get install -y python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install -r requirements.txt --no-cache --break-system-packages



# Environment variables for offline mode
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV NLTK_DATA=/app/nltk_data

# Copy all necessary files and folders
COPY main.py .
COPY xgb_model.pkl .
COPY rf_model.pkl .
COPY tfidf_vectorizer.pkl .
COPY gpt2 ./gpt2
COPY bert-base-uncased ./bert-base-uncased
COPY nltk_data ./nltk_data

# Run the main script
CMD ["python", "main.py"]