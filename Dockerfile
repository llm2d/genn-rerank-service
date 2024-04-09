# Use Python 3.10.13 as the base image
FROM python:3.10.13

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container's working directory
COPY . /app

# Set the pip source to Tsinghua mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the container's entrypoint command
CMD ["python", "main.py"]