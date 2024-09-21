import asyncio
import os
import streamlit as st
import openai

from dotenv import load_dotenv

from langsmith.wrappers import wrap_openai
from langsmith import traceable

# Load environment variables
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
endpoint_url = "https://api.openai.com/v1"

configurations = {
    "mistral_7B_instruct": {
        "endpoint_url": os.getenv("MISTRAL_7B_INSTRUCT_ENDPOINT"),
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "model": "mistralai/Mistral-7B-Instruct-v0.3"
    },
    "mistral_7B": {
        "endpoint_url": os.getenv("MISTRAL_7B_ENDPOINT"),
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "model": "mistralai/Mistral-7B-v0.1"
    },
    "openai_gpt-4": {
        "endpoint_url": os.getenv("OPENAI_ENDPOINT"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o-mini",
    }
}

# Choose configuration
config_key = "openai_gpt-4"
#config_key = "mistral_7B_instruct"
#config_key = "mistral_7B"

# Get selected configuration
config = configurations[config_key]

# Model kwargs
gen_kwargs = {
    "model": config["model"],
    "temperature": 0.3,
    "max_tokens": 500
}

# Initialize the OpenAI async client
client = wrap_openai(openai.AsyncClient(api_key=config["api_key"], base_url=config["endpoint_url"]))

with st.sidebar:
    "[View the source code](https://github.com/anamsarfraz/recallhq)"

st.title("📝 Event Q&A with OpenAI")

@traceable
async def generate_answer():
    uploaded_file = st.file_uploader("Upload a file you want to ask questions about", type=("txt", "md"))
    question = st.text_input(
        "Ask something about the file",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    response_container = st.empty()
    response = "### Answer\n"
    if uploaded_file and question and openai_api_key:
        article = uploaded_file.read().decode()
        prompt = f"""Here's an article:\n\n<article>
        {article}\n\n</article>\n\n{question}"""

        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **gen_kwargs)

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                response += token
                response_container.write(response)

asyncio.run(generate_answer())