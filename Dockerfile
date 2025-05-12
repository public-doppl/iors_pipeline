# Use an official Python runtime as a parent image
FROM python:3.13

# Set the working directory
WORKDIR /app

# Copy the required files to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends jq

COPY src/ /app/src
COPY data/input/ /app/data/input/

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="/app"

# Command to run the application
ENTRYPOINT ["bash","./src/entrypoint.sh"]