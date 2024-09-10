FROM python:3.11

WORKDIR /code

COPY src/fr/main.py /code/

#RUN pip install --no-cache-dir --upgrade git+https://github.com/mangG907/fishmlserv.git@1.1/find_k
