# Docker

```bash
docker build -t draftfly .
docker run --rm draftfly python -m draftfly doctor
docker run --rm -p 8000:8000 draftfly python -m draftfly serve
```

或用 docker-compose：

```bash
docker-compose up
```
