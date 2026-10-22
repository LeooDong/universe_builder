REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON ?= $(REPO_ROOT).venv/bin/python

.PHONY: all install test lint hooks release-check

all: test

install:
	$(PYTHON) -m pip install -e "$(REPO_ROOT)"

test:
	$(PYTHON) -m pytest -c "$(REPO_ROOT)pyproject.toml" "$(REPO_ROOT)tests" -q

release-check:  ## before tagging: the suite and lint
	$(PYTHON) -m pytest -c "$(REPO_ROOT)pyproject.toml" "$(REPO_ROOT)tests" -q
	$(PYTHON) -m flake8 --max-line-length=119 "$(REPO_ROOT)src" "$(REPO_ROOT)tests"

lint:
	$(PYTHON) -m flake8 --max-line-length=119 "$(REPO_ROOT)src" "$(REPO_ROOT)tests"

hooks:  ## install the repo's git hooks (refuses direct commits on main)
	@git config core.hooksPath hooks
	@echo "hooks installed: $$(git config core.hooksPath)"
