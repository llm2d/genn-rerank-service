# Use Python 3.10.13 as the base image
FROM python:3.10.13

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container's working directory
COPY . /app

# Install the project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the container's entrypoint command
CMD ["python", "main.py"]