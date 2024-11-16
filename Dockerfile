FROM python:3.12-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./data /app/data

COPY ./mlruns /app/mlruns

COPY ./src/api /app/src/api
COPY ./src/__init__.py /app/src/__init__.py

COPY ./params.yaml /app/params.yaml

EXPOSE 8000

CMD ["fastapi", "run", "src/api/main.py", "--host", "0.0.0.0", "--port", "8000"]