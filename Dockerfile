FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN python -m compileall -q draftfly

EXPOSE 8000

CMD ["python", "-m", "draftfly", "serve"]
