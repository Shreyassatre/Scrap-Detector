# Use a standard, slim Python 3.11 base image for stability
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install the Python dependencies
# --no-cache-dir makes the image smaller
# --upgrade pip is good practice
RUN pip install --no-cache-dir --upgrade pip -r /code/requirements.txt

# Copy your FastAPI application code into the container
COPY ./api.py /code/api.py

# Expose the port that the app will run on. Hugging Face Spaces expects 7860.
EXPOSE 7860

# Define the command to run your application
# This tells uvicorn to run the 'app' instance from the 'api.py' file
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]