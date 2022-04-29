FROM datamechanics/spark:3.1-latest

ENV PYSPARK_MAJOR_PYTHON_VERSION=3

WORKDIR /opt/wine-pred-app
RUN conda install numpy

RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY Predictions.py .
