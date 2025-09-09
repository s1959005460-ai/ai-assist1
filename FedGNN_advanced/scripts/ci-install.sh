#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi
pip install pre-commit black isort flake8 pytest pytest-cov
