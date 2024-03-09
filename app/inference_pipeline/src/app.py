import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from typing import List
from pathlib import Path
from threading import Thread

from .langchain_bot import FinanceBot
from .utils import create_logger, load_yaml

logger = create_logger("logs/outputs.log")
config = load_yaml(Path("src/config.yml"))


bot = FinanceBot(
        config=config
    )

# === Gradio Interface ===


def predict(message: str, history: List[List[str]], about_me: str):
    """
    Predicts a response to a given message using the financial_bot Gradio UI.

    Args:
        message (str): The message to generate a response for.
        history (List[List[str]]): A list of previous conversations.
        about_me (str): A string describing the user.

    Returns:
        str: The generated response.
    """

    generate_kwargs = {
        "about_me": about_me,
        "question": message,
        "to_load_history": history,
    }

    if bot.is_streaming:
        t = Thread(target=bot.answer, kwargs=generate_kwargs)
        t.start()

        for partial_answer in bot.stream_answer():
            yield partial_answer
    else:
        yield bot.answer(**generate_kwargs)


demo = gr.ChatInterface(
    predict,
    textbox=gr.Textbox(
        placeholder="Ask me a financial question",
        label="Financial question",
        container=False,
        scale=7,
    ),
    additional_inputs=[
        gr.Textbox(
            "I am a student and I have some money that I want to invest.",
            label="About me",
        )
    ],
    title="Your Personal Financial Assistant",
    description="Ask me any financial or crypto market questions, and I will do my best to answer them.",
    theme="soft",
    examples=[
        [
            "What's your opinion on investing in startup companies?",
            "I am a 30 year old graphic designer. I want to invest in something with potential for high returns.",
        ],
        [
            "What's your opinion on investing in AI-related companies?",
            "I'm a 25 year old entrepreneur interested in emerging technologies. \
             I'm willing to take calculated risks for potential high returns.",
        ],
        [
            "Do you think advancements in gene therapy are impacting biotech company valuations?",
            "I'm a 31 year old scientist. I'm curious about the potential of biotech investments.",
        ],
    ],
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)