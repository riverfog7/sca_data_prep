#!/usr/bin/env bash

set -euo pipefail

uv run ruff check --fix src/ scripts/ test/
uv run ruff format src/ scripts/ test/
uv run ty check src/ scripts/
