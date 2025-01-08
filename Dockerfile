FROM python:3.11.4-slim

ENV FLASK_APP=main.py
ENV FLASK_ENV=development

COPY . /app
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1





## Make the DEBUG param to be commented out before pushing it to the production
# ENV DEBUG = True

RUN pip install -r requirements.txt


ENTRYPOINT FLASK_APP=/app/app.py flask run --host=0.0.0.0 --port=80
