# FedGNN_advanced — Tutorial & Quickstart

This tutorial shows how to run a complete experiment end-to-end using the FedGNN_advanced codebase.
It covers environment setup, a quick-run on a small dataset, evaluation, and pointers for reproducible experiments.

---

## 1. Install dependencies

Create a Python virtualenv and install requirements (example):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# for tests:
pip install pytest pytest-asyncio
