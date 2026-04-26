# ── Makefile for 3D Segmentation Pipeline ──────────────────────────────────

.PHONY: install run test clean lint

# Default input data
INPUT_CLOUD ?= data/synthetic/room_01.ply
CONFIG_FILE ?= configs/default.yaml

install:
	pip install -r requirements.txt

run:
	python3 main.py --input $(INPUT_CLOUD) --config $(CONFIG_FILE)

test:
	pytest tests/ --verbose

clean:
	rm -rf outputs/*
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf src/__pycache__

lint:
	flake8 src/ main.py
	mypy src/ main.py
