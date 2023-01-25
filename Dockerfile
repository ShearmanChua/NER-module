FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update
RUN apt-get -y install python3-pip vim git
RUN apt-get -y install libfreetype-dev libfreetype6 libfreetype6-dev

RUN pip install -U pip

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install fastapi pandas requests torch fsspec && pip install "uvicorn[standard]"

RUN mkdir /NER_module && mkdir /NER_module/src && mkdir /NER_module/config && mkdir /NER_module/models && mkdir /NER_module/data
COPY spanNER/src /NER_module/src
WORKDIR /NER_module/src

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

CMD ["/bin/bash"]
