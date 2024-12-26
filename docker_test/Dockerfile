FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
USER root
ENV TZ=Europe/Moscow

RUN apt update && apt upgrade -y

ENV SYSTEM_LANG=ru_RU.UTF-8
RUN apt install -y language-pack-ru tzdata && update-locale LANG=$SYSTEM_LANG

RUN apt install -y sudo bash curl git software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.11 python3.11-venv python3.11-dev python3.11-gdbm
RUN ln -s /usr/bin/python3.11 /usr/bin/python && alias python='/usr/bin/python3.11' && alias python3='/usr/bin/python3.11'
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install --upgrade pip setuptools wheel

RUN apt -y clean && apt -y autoremove && apt -y autoclean && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*



RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

#RUN pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12==24.6.*

# Копирование файла зависимостей в рабочую директорию
COPY requirements.txt /opt
COPY test.py /opt

RUN pip install  -r /opt/requirements.txt

RUN pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.8.*



# Копирование остального кода приложения в контейнер
ENV PYTHONPATH /opt
ENV PYTHONUNBUFFERED=1
# Команда для запуска скрипта
CMD ["python", "/opt/test.py"]