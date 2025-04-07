# Use official Python image as a base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]