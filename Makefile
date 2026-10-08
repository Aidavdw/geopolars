SHELL=/bin/bash

.PHONY: install install-release polars-src run-geo run-line run-mean-coordinate run-mean-coordinate-manual run-crs rebuild test test-rust test-python clean

# The polars commit we build against. Cargo.toml is the source of truth;
# this checks out the same commit for uv to build Python polars from.
POLARS_REV := $(shell sed -n 's/^polars = .*rev = "\([0-9a-f]*\)".*/\1/p' Cargo.toml)
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

# Rust tests can be a bit slower, so only run them if we ask specifically.
test: test-python

test-all: test-rust test-python

## Rust unit tests. 
## Run with --no-default-features  so pyo3's `extension-module` is dropped
test-rust:
	cargo test -p geopolars --no-default-features

## Python tests. Delegates to the package Makefile, which builds first.
test-python:
	@$(MAKE) -s -C geopolars test

## Run the geoarrow.point proof of concept.
run-geo: install
	uv run examples/run_geo.py

## Run the geoarrow.linestring proof of concept.
run-line: install
	uv run examples/run_line.py

## Run the CRS reprojection example, for points and lines.
run-crs: install
	uv run examples/run_crs.py

## Run the mean coordinate proof of concept.
run-mean-coordinate: install
	uv run examples/run_mean_coordinate.py

## Run the same mean coordinate written by hand, in plain Polars.
run-mean-coordinate-manual: install
	uv run examples/run_mean_coordinate_manual.py

clean:
	-@rm -rf .venv target
	-@rm -f uv.lock
