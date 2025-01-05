FROM python:3.11.4-slim

# Set environment variables for Flask
ENV FLASK_APP=main.py
ENV FLASK_ENV=development

# Copy the application code into the container
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose port 5000 to the container
EXPOSE 5000

# Use flask run to start the Flask application, binding it to 0.0.0.0 on port 5000
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
