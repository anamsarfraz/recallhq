import chainlit as cl
import openai
import os
import base64
from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
from video_rag import VideoRag
from ingest_video import Video, generate_random_string, make_tempdirs
import re, json, shutil

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """\
You are a helpful assistant in providing information about events from a Youtube URL that the 
user might provide.

If the user mentions they have a video url of an event or if wish to provide this url,
    generate the function below to get this as input from the user: 

{
    "function_name": "get_youtube_url",
    "rationale": "Explain why would you like to call this function"
}

If you are ready to process a video, generate this function call:
{
    "function_name": "process_video",
    "video_url": "Url of the video if the user provides, one.",
    "rationale": "Explain why would you like to call this function"
}

If the user asks questions regarding the video, generate this function call:
{
    "function_name": "query_video",
    "query": "User's query as a sentence.",
    "rationale": "Explain why would you like to call this function"
}
"""

client = AsyncOpenAI()

KB_BASE = "./events-kb"
VIDEO_FOLDER = "./video_lib/"

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 1500
}

async def generate_llmresponse(client, message_history, gen_kwargs):
    llm_response = await client.chat.completions.create(messages=message_history, stream=False, **gen_kwargs)
    # Extract the assistant's response
    if llm_response and llm_response.choices[0]:
        message_content = llm_response.choices[0].message.content
        return message_content
   
    return None

def extract_json(text):
    # Regular expression to capture JSON-like objects
    json_regex = r'(\{[^{}]*\})'

    # Find the first match
    matches = re.search(json_regex, text)

    if matches:
            # Get the prefix, json string, and postfix
        json_str = matches.group(1)
        prefix = text[:matches.start()]
        postfix = text[matches.end():]
        
        try:
            # Parse the matched JSON string into a Python dictionary
            json_obj = json.loads(json_str)
            print("Prefix:", prefix)
            prefix = prefix.replace("```json", "")
            print("Extracted JSON string:", json_str)
            print("Postfix:", postfix)
            print("Parsed JSON object:", json_obj)
            return (prefix, json_obj, postfix)
        except json.JSONDecodeError:
            print("Matched string is not a valid JSON object")
    else:
        print("No JSON object found")
    return (None, None, None)

async def get_youtube_url():
    res = await cl.AskUserMessage(content="Please provide a Youtube URL:", timeout=30).send()
    if res:
        elements = [
        cl.Video(name="Youtube video", url=res['output'], display="inline"),
         ]
        await cl.Message(
            content=f"Url provided: {res['output']}",  elements=elements
        ).send()
    cl.user_session.set("url", res['output'])
    return res['output']

def add_system_message(message_history, msg):
    message_history.append({"role": "system", "content": msg})

def add_user_message(message_history, msg):
    message_history.append({"role": "user", "content": msg})

def add_assistant_message(message_history, msg):
    message_history.append({"role": "assistant", "content": msg})

def is_http_url(url):
    return "youtube" in url or "http" in url

def download_and_process_video(url, events_folder, output_folder):
    print("url = ", url)
    if is_http_url(url):
        print("is http returned true")
        video = Video.from_url(url)
        video.download(output_folder)
    else:
        # change this for temporary debugging.
        file_fullpath =  os.getcwd() + "/video_lib_2/" + url
        print("Processing local video file: ", file_fullpath)
        make_tempdirs(output_folder)
        # we mimic a youtube download by just moving the file there.
        shutil.copy(file_fullpath, output_folder)
        video = Video.from_file(f"{output_folder}/{url}")

    (v, a, t) = video.process_video_with_index(events_folder)
    # OLD: video.extract_images(f"{images_folder}/images/") 
    video.extract_images_with_index(events_folder)
    print(f"Video saved as {v}, audio as {a}, text as {t}")
    return (v, a, t)

def compute_video_rag(url):
    event_id = generate_random_string(10)
    events_folder = f"{KB_BASE}/event_{event_id}"
    make_tempdirs(events_folder)
    v, a, t = download_and_process_video(url, events_folder, f"{VIDEO_FOLDER}/video_{event_id}/")
    #print(f"Copying text file from {t} to {events_folder}")
    #shutil.copy(t, f"{events_folder}/")
    video_rag = VideoRag(f"{events_folder}/data")
    video_rag.create_index()
    video_rag.init_multimodal_oai()
    cl.user_session.set("video_rag", video_rag)
    print("Processing complete.")

async def process_query(query_str):
    video_rag = cl.user_session.get("video_rag", None)
    if video_rag:
        text, images = video_rag.retrieve(query_str)
        print("Query result:", text)
        print("Images result:", images)
        for img in images:
            print(img.metadata['file_path'])
        image_list = [cl.Image(path=f"{img.metadata['file_path']}", name="images", display="inline") for img in images]
        response = video_rag.query_with_oai(query_str, text, images)
        await cl.Message(content=response, elements=image_list).send()

@cl.on_chat_start
async def start_chat():
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)
    await cl.Message(content=f"Hello, how can i help you ?").send()


@cl.on_message
async def on_message(message: cl.Message):
    # Record the AI's response in the history
    message_history = cl.user_session.get("message_history", [])
    print("Got message: ", message.content)
    add_user_message(message_history, message.content)

    # 1. Generate a response from the LLM
    last_llm_response = await generate_llmresponse(client, message_history, gen_kwargs)

    # 2. Parse the response to see if there is a function call.
    (prefix, json_obj, postfix) = extract_json(last_llm_response)
    if prefix and prefix != "":
        await cl.Message(content=prefix).send()
    function_call =  json_obj if json_obj and "function_name" in json_obj else None
    if function_call:
        #3. Execute function call and request a followup response from the LLM.
        if function_call["function_name"] == "get_youtube_url":
            youtube_url = await get_youtube_url()
            add_system_message(message_history, f"User provided the following URL: f{youtube_url}. Please thank the user and ask if we should proceed with processing this video.")
            last_llm_response = await generate_llmresponse(client, message_history, gen_kwargs)
            print("exit: last_llm_response: ", last_llm_response)
        elif function_call["function_name"] == "process_video":
            if "video_url" in function_call:
                youtube_url = function_call["video_url"]
            else:
                youtube_url = cl.user_session.get('url')
            compute_video_rag(youtube_url)
            add_system_message(message_history, f"Video processing complete. User may ask questions regarding the video.")
            last_llm_response = await generate_llmresponse(client, message_history, gen_kwargs)
        elif function_call["function_name"] == "query_video":
            if "query" in function_call:
                query_str = function_call["query"]
                await process_query(query_str)



    response_message = cl.Message(content=last_llm_response)
    await response_message.send()

    add_assistant_message(message_history, last_llm_response)
