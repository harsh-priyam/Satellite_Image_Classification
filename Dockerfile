FROM python:3.11.4-slim

ENV FLASK_APP=main.py

ENV FLASK_ENV=development

COPY . /app

WORKDIR /app

RUN python3 -m pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python","app.py"]