FROM 3.8.3-alpine3.11

COPY requirements.txt /tmp/custom-requirements.txt

COPY translate.py /tmp/translate.py

RUN pip3 install --upgrade -r /tmp/custom-requirements.txt

CMD ["python", "/tmp/translate.py"]