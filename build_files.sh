#!/bin/bash
# Install pip if not already installed
if ! command -v pip &> /dev/null
then
    echo "pip could not be found, installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
fi

# Install dependencies
pip install -r requirements.txt

# Collect static files
python3.9 manage.py collectstatic --noinput
