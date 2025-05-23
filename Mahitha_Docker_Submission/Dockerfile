# Use official Python base with PyTorch
FROM python:3.10-slim

# Set working directory inside container


# Copy all files into container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
RUN find /app -type f -name "*.jsonl" -delete
#



# Set the default command
CMD ["python", "main.py"]
