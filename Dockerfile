### Dockerfile
FROM python:3.8

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port and run app
EXPOSE 5000
CMD ["python", "app.py"]