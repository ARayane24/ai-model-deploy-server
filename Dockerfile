FROM python:3.11

WORKDIR /project

COPY requirements.txt .
RUN pip install -r requirements.txt
