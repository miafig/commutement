FROM python:3.11

WORKDIR /app

COPY ./backend /app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 7860

CMD ["flask", "--app", "server", "run", "--host=0.0.0.0", "--port=7860"]
