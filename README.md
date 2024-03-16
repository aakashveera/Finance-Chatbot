# FINANCE ASSISTANT

- This repository consists of modules necassary for creating a RAG based finance chatbot with best MLOps practises.

- The chatbot can respond to finance and investment related queries based on the latest finance news using a RAG architecture.


### TOOLS AND FRAMEWORKS USED:

1. <b>Vector DB:</b> Qdrant
2. <b>Embedding Model:</b> sentence-transformers/all-MiniLM-L6-v2
3. <b>LLM model:</b> mistral-7b-instruct-v0.2
4. <b>Experiment Tracking & Model Registry:</b> Comet-ML
5. <b>CI/CD:</b> Github Actions
6. <b>Cloud:</b> AWS EC2, AWS ECR
7. <b>UI:</b> Gradio
8. <b>Other Frameworks:</b> langchain, transformers, bytewax, pytorch, openai, peft


## SYSTEM ARCHITECTURE

- This application follows a three pipeline architecture for creating the chatbot.

 ### 1. FEATURE PIPELINE/ STREAMING PIPELINE

 - This pipeline fetches the finance news in realtime using a external news provider API,

 - Convert the news into a unique embedding using a BERT based model and stores the embedding & metadata into a vector database.

 - This embeddings and news will later be retrieved during training and inferencing.

 - More details and instruction to start this pipeline are provided on the streaming_pipeline module's README file.

 ### 2. TRAINING PIPELINE

 - This pipeline prepares a custom finance QA dataset using knowledge distillation techniques with a large LLM model.

 - Finetunes a open-source LLM model with the prepared dataset, uses comet-ml for experiment tracking and pushes the lora adapter onto comet-ml's model registry.

 - More details and instruction to start this pipeline are provided on the training_pipeline module's README file.

 ### 3. INFERNCE PIPELINE

 - Once feature pipeline is initiated and training pipeline is completed, inferencing pipeline builds the chatbot UI using gradio, initializes the finetuned LLM model, embedding model and connects the Vector DB as a sequential pipeline using langchain.

 - The chain fetches the user query via UI, converts the query into a vector using the embdding model, retrieves the relevant news articles using vector search,

 - Then prepares the prompt using the current query, past chat history & relevant news and later inferes the prompt using the finetuned LLM model.

 - The generated response will be streamed to the user in real time on the frontend.
 
 - Checkout the inference_pipeline folder to know more details and how to start this pipeline.

<br>
<b>REFERENCES:</b>

- https://github.com/iusztinpaul/hands-on-llms