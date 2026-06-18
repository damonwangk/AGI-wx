# Python 版本直接在镜像中安装，避免宿主机架构相关的预编译步骤。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY frontend/ ./frontend/
RUN pip install --no-cache-dir .

EXPOSE 8090

CMD ["agi-assistant"]
