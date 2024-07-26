FROM python:3.12-slim-bookworm
ENV ROLE=client
ENV CONFIG_PATH=/opt/oran/config.ini
RUN mkdir -p /opt/oran
WORKDIR /opt/oran
RUN apt update; apt install -y git vim curl;
COPY requirements.txt /opt/oran/
RUN python3 -m pip install -r requirements.txt

COPY ./src/config.ini /opt/oran/
COPY ./src/*.py /opt/oran/src/

CMD python3 /opt/oran/src/main.py
