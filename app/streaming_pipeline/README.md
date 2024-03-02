## STREAMING PIPELINE
This pipeline fetches the finance news from Alpaca API, Embeds the news into vectore using huggingface transformers, and inserts the vector and metadata into Qdrant Vector DB

## INSTRUCTIONS TO RUN

### 1. RUNNING IN STANDALONE MODE

1. Make sure python3 is installed.
2. Install all the dependencies using `pip install -r requirements.txt`
3. Setup the API KEYS as environment variables using export command.
    - `export QDRANT_API_KEY=<your-api-key>`
    - `export ALPACA_API_KEY=<your-api-key>`
    - `export ALPACA_API_SECRET=<your-api-key>`
4. Run `./run_realtime_dev.sh` to initiate the pipeline in real-time monitoring mode.
4. Run `./run_batch.sh` to initiate the pipeline in batch mode. Modify the start, end dates accordingly in `run_batch.sh` file.


### 2. RUNNING IN A DOCKER CONTAINER

1. Make sure docker and docker-compose is installed.
2. RUN `docker build -t aakashveera/alpaca_stream:v1.0.0 .` to build the docker image.
3. RUN `docker-compose up -d` to start the docker container.