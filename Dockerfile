FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade -r backend/requirements.txt

EXPOSE 7860

CMD ["flask", "--app", "backend/app", "run", "--host=0.0.0.0", "--port=7860"]
