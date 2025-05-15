#!/bin/bash

# Install dependencies (just in case)
pip install --upgrade pip
pip install -r requirements.txt

# Train the model
python app/train.py

# Start API
uvicorn app.main:app --host 0.0.0.0 --port $PORT
web: ./railway-run.sh