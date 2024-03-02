# Use the specified image as the base
FROM python:3.10-slim-bullseye as release

COPY ./requirements.txt ./

# Install Poetry using pip and clear cache
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy the poetry lock file and pyproject.toml file to install dependencies
COPY .src /app
COPY .run_realtime_prod.sh /app

# Give execution permission to your shell script
RUN chmod +x /app/run_realtime_prod.sh

# Run your shell script
CMD ["/app/run_realtime_prod.sh"]