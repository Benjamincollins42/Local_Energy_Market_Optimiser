FROM python:3.9.12-slim-bullseye

COPY ./* ./

RUN pip install -r ./requirements.txt