# Base image (CPU or GPU)
FROM python:3.9-slim


WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build
RUN python -m nltk.downloader punkt

# Copy the inference script
COPY main.py .


COPY xgb_model.pkl .
COPY rf_model.pkl .
COPY tfidf_vectorizer.pkl .

ENTRYPOINT ["python", "main.py"]