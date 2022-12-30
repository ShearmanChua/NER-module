FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN mkdir -p /spanREL
WORKDIR /spanREL

COPY . /spanREL

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# RUN pip install hydra-core --upgrade
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
