# Start from the official Python 3.10 image
FROM python:3.10 

# Set the working directory to /app inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all the project files into the container (including the api folder and model folder)
COPY . /app/

# Expose the app on port 8000
EXPOSE 8000

# Run the FastAPI app using Uvicorn, pointing to the correct location of app.py
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]