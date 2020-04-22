FROM python:3.7-slim

RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean

# install the notebook package
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache notebook && \

# install requirements
COPY requirements.txt /app/requirements.txt
WORKDIR app
RUN pip install --user -r requirements.txt
COPY . /app

# create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
USER ${USER}

RUN pip install -r requirements.txt
RUN pip3 install -r requirements.txt
