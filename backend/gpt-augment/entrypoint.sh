#!/bin/bash
echo "🔍 ENV VARS:"
env

echo "📁 FILES:"
ls -la /gpt-augment

echo "🐍 Python Version:"
python --version

echo "🏃 Starte gpt-paraphrasing.py..."
python gpt-paraphrasing.py