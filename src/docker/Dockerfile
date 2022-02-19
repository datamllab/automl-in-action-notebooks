FROM tensorflow/tensorflow:2.8.0-gpu

COPY requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN apt install -y graphviz
