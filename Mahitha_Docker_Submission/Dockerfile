FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04



COPY requirements.txt .
# ✅ Install Python + pip
RUN apt-get update && \
    apt-get install -y python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install -r requirements.txt --no-cache --break-system-packages

RUN python -m spacy download en_core_web_sm --break-system-packages


# ✅ Set Hugging Face to offline mode
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1


# ✅ Copy code + models
COPY main.py .
COPY xgboost_model.json .
COPY gpt2 ./gpt2
COPY nltk_data ./usr/local/nltk_data



# ✅ Run the script
CMD ["python", "main.py"]