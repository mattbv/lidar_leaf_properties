# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade Cython
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install -r requirements.txt

RUN python3 setup.py install

# Run app.py when the container launches
# CMD ["python", "app.py"]
