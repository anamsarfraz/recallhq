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
        "audio_model": "tts-1"
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
audio_client = wrap_openai(openai.OpenAI(api_key=config["api_key"]))

@traceable
def setup_event_qa():
    with st.sidebar:
        "[View the source code](https://github.com/anamsarfraz/recallhq)"
    with st.sidebar.expander("⚙️ Settings"):
        voice = st.selectbox(
            "Voice Options 🗣️",
            [
                "nova",
                "alloy",
                "echo",
                "fable",
                "onyx",
                "shimmer"
            ],
            help="Choose the voice you want to use. Test out the voices here: https://platform.openai.com/docs/guides/text-to-speech"
        )
    st.title("📝 Event Q&A with OpenAI")

    uploaded_file = st.file_uploader("Upload a file you want to ask questions about", type=("txt", "md"))
    with st.form(key='text_form'):
        question = st.text_area(
            "Ask something about the file",
            placeholder="Can you give me a short summary?",
            disabled=not uploaded_file,
        )
        col1, col2 = st.columns([1,2])
        with col1:
            answer_button = st.form_submit_button(label='Generate Text Response', type="primary")
        with col2:
            audio_button = st.form_submit_button(label='Generate Audio Response🎵', type="primary")
    prompt = ""
    if uploaded_file and question and openai_api_key:
        article = uploaded_file.read().decode()
        prompt = f"""Here's an article:\n\n<article>
        {article}\n\n</article>\n\n{question}"""
    return prompt, voice, answer_button, audio_button


@traceable
async def generate_answer(prompt):
    response_container = st.empty()
    response = ""

    stream = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        **gen_kwargs)

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            response += token
            response_container.markdown(response)
@traceable
def generate_audio(prompt, voice):
    with st.spinner('Generating audio...'):
        text_response = audio_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **gen_kwargs)
        print(text_response.choices[0].message.content)
        audio_response = audio_client.audio.speech.create(
            model=config["audio_model"],
            voice=voice,
            input=text_response.choices[0].message.content
        )
        audio_response.write_to_file("output.mp3")
    with open("output.mp3", "rb") as audio_file:
        st.audio(audio_file, format='audio/mp3')

prompt, voice, answer_button, audio_button = setup_event_qa()
if prompt and answer_button:
    asyncio.run(generate_answer(prompt))
elif prompt and audio_button:
    generate_audio(prompt, voice)