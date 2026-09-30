export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

.PHONY: build up run campaign down test

build:
	docker compose build

up:
	mkdir -p runs data
	docker compose up -d mlflow

run: up
	docker compose run --rm fedroad --out runs/docker --name docker

campaign: up
	docker compose run --rm --entrypoint fedroad-campaign fedroad

down:
	docker compose down

test:
	python -m pytest -q
