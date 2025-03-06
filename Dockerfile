FROM pklehre/ec2025-lab2

# Copy your script into the container
COPY ixa287.py /bin/ixa287.py

# Ensure required dependencies are installed (uncomment if needed)
# RUN apt-get update && apt-get install -y python3-pip

# Run the script correctly
CMD ["-username", "ixa287", "-submission", "python3 /bin/ixa287.py"]