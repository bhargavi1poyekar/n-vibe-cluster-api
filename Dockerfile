FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Create a virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Run
CMD ["python", "main.py"]