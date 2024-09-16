FROM python:3.12-slim-bookworm
RUN apt update && apt install -y git vim curl

RUN mkdir -p /opt/oran/
WORKDIR /opt/oran
ENV DATA_PATH=/opt/oran/
COPY requirements.txt /opt/oran/
RUN python3 -m pip install -r requirements.txt

COPY src/ /opt/oran/