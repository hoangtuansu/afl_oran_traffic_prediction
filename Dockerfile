FROM python:3.9-slim-bookworm
ENV FL_ROLE
WORKDIR /opt

RUN apt update; apt install -y git;

RUN python3 -m pip install -r requirements.txt

RUN mkdir -p /opt/oran
COPY . /opt/oran


CMD ["python3", "/opt/oran/$FL_ROLE.py"]
