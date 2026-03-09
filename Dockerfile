# Use a lightweight Python image (Version 3.11 is very stable)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for pandas
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot script
COPY bot.py .

# Command to run the bot
CMD ["python", "bot.py"]