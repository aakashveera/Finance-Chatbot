import os
import time
import openai
import json
from tqdm import tqdm
from typing import Dict, List

from src.utils import create_logger

INPUT_PATH = "data/input_queries.json"
DATA_DIR = "data/"

logger = create_logger("logs/outputs.log")
    
try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except:
    raise ValueError("Please set OPENAI_API_KEY as a environment variable.")


def load_input_dataset()->List:
    """Load the prepared input dataset that has the short user description and a relevant context fetched from Vector DB.

    Returns:
        List: A list containing a set of dictionary items each with a short description and a relevant context.
    """
    
    json_list = json.load(open(INPUT_PATH))
    
    return json_list

def dump_dataset(json_list: List, filename:str):
    """Dump the prepared training dataset on the specified data directory.

    Args:
        json_list (List): A list containing the dataitems. 
        filename (str): filename to be saved.
    """
    
    with open(DATA_DIR + filename+'.json', "w") as f:
        json.dump(json_list, f, indent=4)


def build_prompt(example: Dict) -> str:
    """Get a prompt for GPT's autocompletion API based on the about_me and context provided.

    Args:
        example (Dict): A dict containing 'about_me' and 'context' items.

    Returns:
        str: Prompt that can be passed to the GPT's chat APIs.
    """
    
    PROMPT_TEMPLATE = """
You are an expert in the stock and crypto markets. I will give you some information about myself and you will provide me with good investment advice.

# ABOUT ME
{ABOUT_ME}

# CONTEXT
{CONTEXT}

Please provide concrete advice in less than 100 tokens, and justify your answer based on the news provided in the context.
"""
    
    
    return PROMPT_TEMPLATE.format(
        ABOUT_ME=example["about_me"],
        CONTEXT=example["context"],
    )
    
def get_response(prompt: str)-> str:
    """_summary_

    Args:
        prompt (str): _description_

    Returns:
        str: _description_
    """

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
    )
    
    try:
        return response["choices"][0]["text"]
    except:
        raise ValueError(f"Error receiving response from GPT. Status Code{response.status_code}")


def main():
    """Main function to load the queries, generate response and save the response along with queries as the training data to the LLM."""
    
    logger.info("Loading query dataset")
    query_list = load_input_dataset()
    
    logger.info("Generating response for the queries")
    output = []
    
    for example in tqdm(query_list):
        prompt = build_prompt(example)
        
        start_time = time.time()
        response = get_response(prompt)
        response_time = time.time() - start_time
        
        lis = example['about_me'].split('\n')
        example['about_me'] = '\n'.join(lis[:-1])
        example['question'] = lis[-1]
        
        if response_time<20: #Free tier can process only 3 requests per minute. So sleep for 20 seconds after processing each request.
            time.sleep(20-(response_time))

        output.append({**example, "response": response})

    logger.info("Dumped the prepared training dataset.")
    train_data, test_data = output[:82], output[82:]
    dump_dataset(train_data,'training_data_v2')
    dump_dataset(test_data,'test_data_v2')
    
    
if __name__ == "__main__":
    main()