FROM fokkodriesprong/docker-pyspark

WORKDIR /home/ubuntu/CS643-AWS-ProgAssgn-2

COPY . /home/ubuntu/CS643-AWS-ProgAssgn-2

RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python", "Predictions.py"]
