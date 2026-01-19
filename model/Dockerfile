FROM python:3.11-slim

RUN apt-get update
RUN apt-get -y install zip unzip

COPY requirements.txt .
COPY paper.py .
COPY embed.py .
COPY helpers.py .
COPY run.sh .

# Create directory for ChromaDB persistence
RUN mkdir -p chroma_db

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD sh run.sh