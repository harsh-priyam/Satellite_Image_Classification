# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install dependencies (add --no-cache-dir to reduce image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application to the container
COPY . .

# Expose the port your app runs on (optional but recommended)
EXPOSE 8080

# Command to run the application (use Gunicorn for production)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
