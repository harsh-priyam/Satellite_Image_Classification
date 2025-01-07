# Dockerfile
# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file if you have dependencies
COPY requirements.txt requirements.txt

# Install any dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Expose port 8080
EXPOSE 8080

# Command to run the application
CMD [ "python", "app.py" ]
