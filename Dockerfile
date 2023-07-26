# Use the official Python image as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app's files to the container
COPY . /app/

# Start the Streamlit app
CMD ["streamlit", "run", "emp_promo_streamlitapp.py"]
