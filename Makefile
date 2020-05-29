# Makefile for launching common tasks

PYTHON ?= python
OUROBOROS_OPTS ?= \
	-e DISPLAY=${DISPLAY} \
	-v ${PWD}:${WORKSPACE} \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /data:/data \
	-v ~/.torch:/root/.torch \
	-v /dev/shm:/dev/shm \
	-v /mnt/fsx:/mnt/fsx \
	-v /mnt/parallel:/mnt/parallel \
	-v /root/.ssh:/root/.ssh \
	-v /var/run/docker.sock:/var/run/docker.sock \
	--network=host \
	--privileged
PACKAGE_NAME ?= detr
WORKSPACE ?= /workspace/$(PACKAGE_NAME)
DOCKER_IMAGE_NAME ?= $(PACKAGE_NAME)
DOCKER_IMAGE ?= $(DOCKER_IMAGE_NAME):latest
DOCKER_ECR_IMAGE ?= 929292782238.dkr.ecr.us-east-1.amazonaws.com/detr:master-latest
DOCKER_LOGIN := "eval $$\( aws ecr get-login --registry-ids 929292782238 --no-include-email --region us-east-1 \)"
NGPUS := $(shell nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
MPIARGS ?= mpirun --allow-run-as-root \
	--map-by socket:OVERSUBSCRIBE \
	-np $(NGPUS) \
	-x MASTER_ADDR=127.0.0.1 \
	-x MASTER_PORT=23455

all: clean test

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

clean-logs:
	find . -name "tensorboardx" | xargs rm -rf && \
	find . -name "wandb" | xargs rm -rf

# Docker Utilities
docker-login:
	@eval $(DOCKER_LOGIN)

docker-build:
	docker build --build-arg AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
		--build-arg AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
		--build-arg AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
		--build-arg WANDB_ENTITY=${WANDB_ENTITY} \
		--build-arg WANDB_API_KEY=${WANDB_API_KEY} \
		-f docker/Dockerfile \
		-t ${DOCKER_IMAGE} .

docker-run:
	nvidia-docker run --name detr --rm \
		-it \
		${OUROBOROS_OPTS} \
		${DOCKER_IMAGE} bash -c "${MPIARGS} ${COMMAND}"

docker-start:
	nvidia-docker run --name detr --rm \
		-d \
		-it \
		${OUROBOROS_OPTS} \
		${DOCKER_IMAGE} && \
		nvidia-docker exec -it detr bash

dist-run:
	nvidia-docker run --name detr --rm \
		${OUROBOROS_OPTS} \
		${DOCKER_IMAGE} \
		${COMMAND}

dist-sweep:
	nvidia-docker run --name detr --rm \
        -e DISPLAY=${DISPLAY} \
        -e HOST=${HOST} \
        -e WORLD_SIZE=${WORLD_SIZE} \
        -e WANDB_PROJECT=${WANDB_PROJECT} \
        -e WANDB_ENTITY=${WANDB_ENTITY} \
        -e WANDB_API_KEY=${WANDB_API_KEY} \
        -e WANDB_AGENT_REPORT_INTERVAL=300 \
        -v ${PWD}:${WORKSPACE} \
        -v ~/.torch:/root/.torch \
        ${OUROBOROS_OPTS} \
        ${DOCKER_IMAGE} \
        ${COMMAND}
