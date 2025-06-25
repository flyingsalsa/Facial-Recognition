FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /app


COPY --link requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN --mount=from=Faces,target=/faces 

COPY src ./src
EXPOSE 5000

CMD ["python", "app.py"]

