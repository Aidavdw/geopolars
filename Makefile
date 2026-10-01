SHELL=/bin/bash

.PHONY: install install-release polars-src run run-geo run-line run-centroid run-centroid-manual run-release rebuild test test-rust test-python clean

# The polars commit we build against, for both Cargo and uv. 
# Single source of truth.
POLARS_REV ?= 8a87616112e5c5ee8b862c675bed3cbaa1e202db
POLARS_DIR := .polars
POLARS_STAMP := $(POLARS_DIR)/.rev-$(POLARS_REV)

# uv only notices a path source changed if its pyproject.toml did,
# so after a new checkout force it to rebuild polars.
UV_SYNC = if [ -e $(POLARS_DIR)/.reinstall ]; then \
		reinstall="--reinstall-package polars --reinstall-package polars-runtime-32"; \
	fi; \
	uv sync $$reinstall && rm -f $(POLARS_DIR)/.reinstall

# Sync the venv, building the plugin unoptimized (fast edit-compile loop).
# this builds polars as debug.
install: polars-src
	export MATURIN_PEP517_ARGS="--profile dev"; $(UV_SYNC)

# Sync the venv, building the plugin optimized.
install-release: polars-src
	$(UV_SYNC)

## Check out POLARS_REV into POLARS_DIR, if it isn't already.
polars-src: $(POLARS_STAMP)

# Fetches only the one commit, not the whole polars history.
$(POLARS_STAMP):
	@mkdir -p $(POLARS_DIR)
	@[ -d $(POLARS_DIR)/.git ] || git -C $(POLARS_DIR) init -q
	git -C $(POLARS_DIR) fetch --depth 1 https://github.com/pola-rs/polars.git $(POLARS_REV)
	git -C $(POLARS_DIR) checkout -q --force FETCH_HEAD
	@rm -f $(POLARS_DIR)/.rev-*
	@touch $(POLARS_DIR)/.reinstall $@

## Force a rebuild of just the plugin, ignoring uv's cache.
rebuild: polars-src
	uv sync --reinstall-package geopolars

test: test-rust test-python

## Rust unit tests. 
## Run with --no-default-features  so pyo3's `extension-module` is dropped
test-rust: polars-src
	cargo test -p geopolars --no-default-features

## Python tests. Delegates to the package Makefile, which builds first.
test-python:
	@$(MAKE) -s -C geopolars test

run: install
	uv run examples/run.py

## Run the geoarrow.point proof of concept.
run-geo: install
	uv run examples/run_geo.py

## Run the geoarrow.linestring proof of concept.
run-line: install
	uv run examples/run_line.py

## Run the coordinate centroid proof of concept.
run-centroid: install
	uv run examples/run_centroid.py

## Run the same centroid written by hand, in plain Polars.
run-centroid-manual: install
	uv run examples/run_centroid_manual.py

clean:
	-@rm -rf .venv target
	-@rm -f uv.lock
