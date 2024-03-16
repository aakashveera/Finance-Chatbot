## INFERENCE PIPELINE

The Inference Pipeline connects the QDRANT DB, finetuned LLM model and embedding model using langchain and builds a chat application with Gradio. 

The entire workflow of the pipeline consists of the following steps,

- Loads Embedding model, LLM model and finetuned lora adapter from model-registry, creates a qdrant client and builds a sequential chain using langchain to process the user queries.
- Once initialized, upon receiving queries, the chain embeds the user's query using the embedding model, fetch the documents relevant to the query using vector search from QDRANT DB,
- Prepares a prompt using the query, fetched news document and the past chat_history for inferencing.
- Infers the prompt using the LLM model and passes the generated response to the gradio UI client.

## INSTRUCTIONS TO RUN LOCALLY
[Note: You will need a GPU with atleast 16GB RAM to load the LLM model with QLoRA]

#### 1. SETTING UP REQUIREMENTS

- Install all the dependencies using `pip install -r requirements.txt`.
- We will be fetching the news documents from Qdrant DB and loading the finetuned lora adapter from comet model registry. 
- So export the necassary environment variables QDRANT_API_KEY, COMET_WORKSPACE, COMET_PROJECT_NAME, COMET_API_KEY. 
- Update the DB URL in the `src/constants.py` file.

#### 2. RUNNING THE APPLICATION

- Initiate the application using the command `python -m src.app`.
- You can now access your financial assistant running at `http://localhost:7860`. You can modify the port number in `src/constants.py` file if necassary.
