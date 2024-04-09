FROM python:3.10-slim

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY model.h5 /app/
COPY inference.py /app

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "inference:app"]
