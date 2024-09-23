import streamlit as st
import time

from constants import KNOWLEDGE_BASE_PATH
from utils import load_state
from rags.text_rag import search_knowledge_base


# Initialize session state for the current app phase
if "phase" not in st.session_state:
    st.session_state.phase = "starters"  # The initial phase is the starter prompts
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history

# Function to generate a response from OpenAI GPT-3.5
def get_openai_response(user_query):
    st.session_state.messages.append({"role": "user", "content": user_query})
    response = search_knowledge_base(user_query, st.session_state.media_label)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    return response

# Function to switch to the chat interface
def switch_to_chat():
    st.session_state.phase = "chat"

# Function to switch back to starter prompts
def switch_to_starters():
    st.session_state.phase = "starters"
    st.session_state.messages = []  # Optionally clear the chat history when going back

def update_chat_history(prompt):
    st.session_state["messages"] = [{"role": "assistant", "content": f"How can I help you answer your questions about \"{prompt}\"?"}]

# Display the current chat history in a chat-like format
def display_chat_history():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# Streamlit layout
st.title("Knowledge Base for Events")

# PHASE: Starter Prompts
if st.session_state.phase == "starters":
    if st.session_state.setdefault("knowledge_base", load_state(KNOWLEDGE_BASE_PATH)):
        starter_prompts = []
        st.write("Chat with one of the events below to get more information.")
        for media_label in st.session_state.knowledge_base.keys():
            starter_prompts.append(media_label)
        # Display starter prompts as buttons
        cols = st.columns(len(starter_prompts))
        for idx, prompt in enumerate(starter_prompts):
            with cols[idx]:
                if st.button(prompt):
                    #get_openai_response(prompt)  # Trigger LLM response
                    st.session_state["media_label"] = prompt
                    update_chat_history(prompt)
                    switch_to_chat()  # Switch to the chat phase
                    st.rerun()  # Rerun the app to update the interface
    else:
        st.markdown("No events found in the knowledge base. Please head over to [Media Processor](/Media_Processor) to add one.")



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
        get_openai_response(user_input)
        st.rerun()  # Update the chat with the new message

    # Button to go back to starter prompts
    if go_back_button:
        switch_to_starters()
        st.rerun()