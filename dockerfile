FROM python:3.12-slim-bookworm AS deploy

ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /app


COPY --link requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN --mount=from=Faces,target=/faces 

COPY src ./src
EXPOSE 5000

CMD ["python", "app.py"]

#figure this out later, this JETSON ORIN has it's own image anyways