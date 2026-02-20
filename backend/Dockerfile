FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 7860

CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
