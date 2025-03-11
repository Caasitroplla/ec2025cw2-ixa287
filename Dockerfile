FROM pklehre/ec2025-lab2

# Copy your script into the container
COPY ixa287.py /bin/ixa287.py

# Run the script correctly
CMD ["-username", "ixa287", "-submission", "python3 /bin/ixa287.py"]