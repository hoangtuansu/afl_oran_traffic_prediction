FROM python:3.12-slim-bookworm as build
RUN apt update; apt install -y git vim curl;
WORKDIR /opt
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN git clone https://github.com/nimbus-gateway/cords-semantics-lib.git
RUN cd cords-semantics-lib && /root/.local/bin/poetry build
#--------
FROM python:3.12-slim-bookworm
ENV MLFLOW_TRACKING_URI=http://10.180.113.115:32256
RUN apt update
COPY --from=build /opt/cords-semantics-lib/dist/cords_semantics-0.2.1.tar.gz /opt

RUN mkdir -p /opt/oran
WORKDIR /opt/oran
COPY requirements.txt /opt/oran/
RUN python3 -m pip install -r requirements.txt

COPY ./src/ /opt/oran/

RUN apt install -y git vim curl

CMD python3 /opt/oran/src/main.py
