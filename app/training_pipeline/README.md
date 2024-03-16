## TRAINING PIPELINE
This pipeline is used to finetune a LLM model using a custom Q&A dataset so that the model can provide more relevant response. The entire workflow of this pipeline consists of following steps,

- Prepare a custom finance QA dataset using OpenAI's GPT model (more details are provided below).
- Fine tune a open-sourced LLM model from hugginface using the prepared dataset (we will be using mistral-7b-instruct-v0.2).
- Log all the experiment & training details on Comet-ML's experiment tracker.
- Store the best model on the Comet-ML's model registry which can be served later for inference.

## INSTRUCTIONS TO RUN LOCALLY
[Note: You will need a GPU with 16GB RAM to load and finetune the LLM model with QLoRA]

#### 1. SETTING UP REQUIREMENTS

- Install all the dependencies using `pip install -r requirements.txt`.
- Login to [comet.com](www.comet.com) and create a project and a API key which we will be using in this pipeline.
- Export the environment variables COMET_WORKSPACE, COMET_PROJECT_NAME, COMET_API_KEY. (COMET_WORKSPACE is your comet username). Comet uses this environment variables to log our experiment details and files.
- Login to [platform.openai.com](https://platform.openai.com/) and create a API key using which we will prepare the QA dataset.

#### 2. PREPARING THE DATASET

- Every parameter, hyperparameter, input & output paths are available in `src/config.yml`. Change if necassary.
- Run the command `python -m src.data.data_generate` to initiate data preparation. It prepares the training data using openAI's gpt-3.5-turbo-instruct model. 
- Note: This part can be skipped if necassary as the prepared data is already available under `data/` directory.

#### 3. TRAINING & INFERENCNG

- Initiate the finetuning using command `python -m src.train`. It initiates the training and will the log the experiment-details, best-model's LORA adapter onto comet-ml's project directory.
- Navigate to the project in comet-ml website register the model onto the model registry with the suitable name and version. We will be using the finetuned adapter for all sort inferencing only from the model registry.
- Initiate validation using the command `python -m src.infer --model_name <comet-username>/<model-name>:<model-version>`. The validation results and generated responses will be stored on the specified output folder on the config file.
