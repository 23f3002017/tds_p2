#!/bin/bash
set -e

# Update pip
pip install --upgrade pip

# Install dependencies with specific flags for pandas
pip install --no-cache-dir \
  Flask==2.3.3 \
  playwright==1.40.0 \
  python-dotenv==1.0.0 \
  requests==2.31.0 \
  gunicorn==21.2.0 \
  pandas==2.0.3

# Install playwright browsers
playwright install chromium