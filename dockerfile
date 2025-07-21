FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.7.21 /uv /uvx /bin/

WORKDIR /app

COPY . .

RUN uv sync --locked

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "/app/app:app", "--host", "0.0.0.0", "--port", "8000"]

