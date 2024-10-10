import streamlit as st
import json
import os
import time
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from constants import KNOWLEDGE_BASE_PATH
from recall_utils import load_state, generate_videoclips
from rags.text_rag import search_knowledge_base, create_new_index, get_llm_response
from rags.scraper import perform_web_search

# CSS for custom styling
st.markdown("""
    <style>
    .stButton > button {
        border: 2px solid transparent; /* Set border to transparent */
        background-color: transparent; /* Set background color to transparent */
        color: #466ea1; /* Set text color to blue */
        font-size: 20px;
        font-weight: bold;
        padding: 0 0; /* Add padding for better appearance */
        border-radius: 5px; /* Optional: round the corners */
        cursor: pointer; /* Change cursor to pointer on hover */
        transition: background-color 0.3s ease, transform 0.1s ease; /* Transition for smooth effects */
        margin: 0 0;
    }
    .stButton > button:hover {
        background-color: #E0F7FA; /* Optional: add a hover effect */
        color: #273d5a; /* Change text color on hover */
        padding: 10px 20px;
        border: 2px solid transparent; /* Set border to transparent */
    }
    .stButton > button:active {
        background-color: #E0F7FA; /* Optional: add a hover effect */
        color: #273d5a; /* Change text color on click */
        border: 2px solid transparent; /* Set border to transparent */
    }
    .app-title {
        color: #466ea1;  /* Updated app title color */
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .app-tags {
        display: inline-block;
        background-color: #e0e0e0;
        color: black;
        padding: 3px 6px;
        margin-right: 5px;
        margin-bottom: 20px;
        border-radius: 5px;
        font-size: 12px;
    }
    .app-tags-container {
        margin-top: -10px;
        margin-bottom: 30px;
    }
    hr {
        margin: 0px;
        border: 2px solid #32A9F1;  /* Updated horizontal line style */
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for the current app phase
if "phase" not in st.session_state:
    st.session_state.phase = "starters"  # The initial phase is the starter prompts
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = load_state(KNOWLEDGE_BASE_PATH)
if "indexes" not in st.session_state:
    st.session_state.indexes = {}
if "futures" not in st.session_state:
    st.session_state.futures = {}

for media_label in st.session_state.knowledge_base.keys():
    if media_label not in st.session_state.indexes and media_label not in st.session_state.futures:
        print(f"Index for {media_label} does not exist. Need to add to futures")
        tp_executor = ThreadPoolExecutor(max_workers=1)
        future = tp_executor.submit(create_new_index, media_label)
        st.session_state.futures[media_label] = [future, tp_executor]
    else:
        print(f"Index for {media_label} exists in future or index")
    
# Function to generate a response from OpenAI GPT-3.5
async def get_openai_response(user_query):
    print(f"User query: {user_query}")
    msg = {"role": "user", "content": user_query}
    st.session_state.messages.append(msg)
    st.chat_message(msg["role"]).write(msg["content"])
    response_container = st.empty()
    img_docs, text_docs = search_knowledge_base(user_query, st.session_state.media_label, st.session_state.indexes)
    prompt = f"""
        Context:
        {text_docs}
        """
    st.session_state.messages.append({"role": "system", "content": prompt})
    # Setting tools call to False to not return function data
    response_text, function_data = await get_llm_response(user_query, messages=st.session_state.messages, tools_call=False, response_container=response_container)
    if response_text:
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    if function_data:
        tp_executor = ThreadPoolExecutor(max_workers=len(function_data))
        futures = []
        for index, index_data in function_data.items():
            function_name = index_data["name"]
            if arguments := index_data["arguments"]:
                arguments = json.loads(arguments)
            print("Function name: ", function_name)
            print("Arguments: ", arguments)
            if function_name == "perform_web_search":
                futures.append(tp_executor.submit(perform_web_search, arguments["query"], arguments["media_label"]))
                print("Web search results: added to threads")
            else:
                print("No function found in the response")
        try:
            response_container.markdown("Searching the web for more information...")
            web_search_results = 'Context: ' + '\n'.join([future.result() for future in futures])
            st.session_state.messages.append({"role": "system", "content": web_search_results})
            print("Web search results: added to message history")
        except Exception as e:
            print(f"Error performing web search: {e}")
        tp_executor.shutdown()
        response_text, function_data = await get_llm_response(user_query, messages=st.session_state.messages, tools_call=False, response_container=response_container)
        print("Response text in the if: ", response_text)
        print("Function data in the if: ", function_data)
        if response_text:
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "This is all the information I could gather for your question."})
    
    if img_docs:
        for doc in img_docs:
            relative_img_path = doc.metadata["file_path"].split('events_kb/', 1)[1]
            new_img_path = os.path.join('events_kb', relative_img_path)
            st.image(new_img_path)
            st.session_state.messages.append({"role": "assistant", "content": new_img_path, "is_image": True})

    if text_docs:
        new_video_path = './temp/video_clips'
        for doc in text_docs:
            text_path = doc['file_path']
            video_path = os.path.join(os.getcwd(), 'temp', 'video_data', Path(text_path).parent.name+'.mp4')
            start_time = doc['timestamps'][0][0]
            end_time = doc['timestamps'][-1][-1]
            print(f"Adding video: {video_path} from {start_time} to {end_time}")

            video_data = [{'video_file': video_path, 'timestamps': [start_time, end_time]}]
            clips, clip_paths = generate_videoclips(new_video_path, video_data)
            st.video(clip_paths[0])
            st.session_state.messages.append(
                {"role": "assistant", "content": clip_paths[0], "is_video": True, "start_time": start_time, "end_time": end_time}
                )
    return response_text

# Function to switch to the chat interface
def switch_to_chat():
    print("Switching to chat on button click")
    st.session_state.phase = "chat"
    if st.session_state["media_label"] in st.session_state.futures:
        print(f"Processing index update for {st.session_state['media_label']}")
        future, tp_executor = st.session_state.futures.pop(st.session_state['media_label'])
        st.session_state.indexes[st.session_state["media_label"]] = future.result()
        tp_executor.shutdown()

# Function to switch back to starter prompts
def switch_to_starters():
    st.session_state.phase = "starters"
    st.session_state.messages = []  # Optionally clear the chat history when going back

def update_chat_history(topic):
    system_prompt = f"You are a helpful assistant that helps people answer questions about {topic}."
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"How can I help you answer your questions about \"{topic}\"?"}
    ]

# Display the current chat history in a chat-like format
def display_chat_history():
    for msg in st.session_state.messages:
        if msg["role"] in {"user", "assistant"}:
            if msg.get("is_image"):
                st.image(msg["content"])
            elif msg.get("is_video"):
                st.video(msg["content"], start_time=msg["start_time"], end_time=msg["end_time"])
            else:    
                st.chat_message(msg["role"]).write(msg["content"])

# Streamlit layout
st.title("Knowledge Base for Events")

# PHASE: Starter Prompts
if st.session_state.phase == "starters":

    if st.session_state.setdefault("knowledge_base", load_state(KNOWLEDGE_BASE_PATH)):
        starter_prompts = []
        #st.write("Chat with one of the events below to get more information about the event.")
        for media_label, event_data in st.session_state.knowledge_base.items():
            if event_data.get("title_image"):
                image_path = os.path.join(os.getcwd(), event_data.get("title_image"))
            else:
                image_path = f"https://via.placeholder.com/150?text={media_label.replace(' ', '+')}"
            starter_prompts.append({
                "title": media_label,
                "tags": event_data["tags"],
                "image": image_path 
            })
        # Extract all unique tags from the events data
        all_tags = sorted(set(tag for event in starter_prompts for tag in event['tags']))

                # Multi-select search bar with pre-filled tags
        selected_tags = st.multiselect("Select filter(s) below to get the related events for your search", all_tags, default=[])

        # Filter the events based on the selected tags
        if selected_tags:
            filtered_events = [event for event in starter_prompts if any(tag in event["tags"] for tag in selected_tags)]
        else:
            filtered_events = starter_prompts

        subtitle = "Query Results" if selected_tags else "Events Information"
        res_suffix = "result" if selected_tags else "event"
    
        st.markdown(f'<h3 class="query-results-title">{subtitle}</h3>', unsafe_allow_html=True)
        st.markdown('<h7 class="query-results-title">Chat with one of the events below to get more information about the event.</h7>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)

        # Display number of results in a highlighted tag style
        st.markdown(f'<span style="background-color:#E0F7FA; color:black; padding:3px 10px; border-radius:5px;">{len(filtered_events)} {res_suffix}(s)</span>', unsafe_allow_html=True)

        # Display filtered events
        cols = st.columns(3)  # Set up a multi-column layout

        for i, event in enumerate(filtered_events):
            with cols[i % 3]:
                st.image(event["image"], use_column_width=True)
                #st.markdown(f'<h5 class="app-title">{event["title"]}</h5>', unsafe_allow_html=True)
                if st.button(f"{event['title']}", key=event['title']):
                    #get_openai_response(prompt)  # Trigger LLM response
                    st.session_state["media_label"] = event['title']
                    update_chat_history(event['title'])
                    switch_to_chat()  # Switch to the chat phase
                    st.rerun()  # Rerun the app to update the interface
                tags_html = " ".join([f'<span class="app-tags">{tag}</span>' for tag in event["tags"]])
                st.markdown(f'<div class="app-tags-container">{tags_html}</div>', unsafe_allow_html=True)

        # If no results found
        if not filtered_events:
            st.warning("No events found matching your search.")
    else:
        st.warning("No events found in the knowledge base. Please head over to [Media Processor](/Media_Processor) to add one.")



# PHASE: Chat Interface
if st.session_state.phase == "chat":
    # Display chat history
    go_back_button = st.button("Go Back to Knowledge Base")
    display_chat_history()
    # Free-form input for chat
    # user_input = st.text_input("Ask your own question:")
    
    # Button to send message
    #if st.button("Send"):
    if user_input := st.chat_input():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(get_openai_response(user_input))
        # st.rerun()  # Update the chat with the new message

    # Button to go back to starter prompts
    if go_back_button:
        switch_to_starters()
        st.rerun()