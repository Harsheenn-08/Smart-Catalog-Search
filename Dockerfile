# Step 1: Base Python environment
FROM python:3.10-slim

# Step 2: Set container workspace directory
WORKDIR /app

# Step 3: Install C++ build tools required by FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy dependency list and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy application code into container
COPY . .

# Step 6: Expose port for FastAPI server
EXPOSE 8000

# Step 7: Launch Uvicorn server on container startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]