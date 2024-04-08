FROM python:3.11.6

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY models /app/models
COPY inference.py /app

EXPOSE 5000

CMD ["python", "inference.py"]