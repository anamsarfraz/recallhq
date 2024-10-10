import asyncio
import base64
import json
import threading
from asyncio import run_coroutine_threadsafe

import numpy as np
import sounddevice as sd
import streamlit as st


import os
import queue
import tzlocal
from datetime import datetime

import websockets
from dotenv import load_dotenv

load_dotenv()

AUTOSCROLL_SCRIPT = """
<script>
    let lastScrollHeight = 0;
    let userHasScrolledUp = false;

    function autoScroll() {
        const streamlitDoc = window.parent.document;
        const textArea = streamlitDoc.getElementsByClassName('st-key-logs_container')[0];
        const scrollArea = textArea.parentElement.parentElement;
        
        // Check if content height has changed
        if (scrollArea.scrollHeight !== lastScrollHeight) {
            // Only auto-scroll if user hasn't scrolled up
            if (!userHasScrolledUp) {
                scrollArea.scrollTop = scrollArea.scrollHeight;
            }
            lastScrollHeight = scrollArea.scrollHeight;
        }

        // Detect if user has scrolled up
        const isScrolledToBottom = scrollArea.scrollHeight - scrollArea.scrollTop <= scrollArea.clientHeight + 50; // 50px threshold
        userHasScrolledUp = !isScrolledToBottom;
    }

    // Run auto-scroll check periodically
    setInterval(autoScroll, 500);
</script>
"""

HIDE_STREAMLIT_RUNNING_MAN_SCRIPT = """
<style>
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
</style>
"""

OAI_LOGO_URL = "https://raw.githubusercontent.com/openai/openai-realtime-console/refs/heads/main/public/openai-logomark.svg"

EVENT_1_JSON = """
```
{
    "type": "conversation.item.create",
    "item": {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "This is the way the world ends..."
            }
        ]
    }
}
```
"""

EVENT_2_JSON = """
```
{
    "type": "response.create"
}
```
"""

EVENT_3_JSON = """
```
{
    "type": "session.update",
    "session": {
        "voice": "echo",
        "instructions": "Always answer like an angry pirate."
    }
}
```
"""

DOCS = f"""
First, make sure that your OpenAI API key is set in the environment variable `OPENAI_API_KEY`.  Then click the `Connect` button.
Send raw json event payloads by pasting them in the input text area and clicking `Send`.  You should then see events streaming in the logs area.
As a test, trying sending:

{EVENT_1_JSON}

followed by:

{EVENT_2_JSON}

Or, to change the voice or instructions, run this at the start of a session:

{EVENT_3_JSON}

You can find the OpenAI realtime events documented [here](https://platform.openai.com/docs/guides/realtime/events).
"""

st.set_page_config(layout="wide")

audio_buffer = np.array([], dtype=np.int16)

buffer_lock = threading.Lock()

if "audio_stream_started" not in st.session_state:
    st.session_state.audio_stream_started = False

def audio_buffer_cb(pcm_audio_chunk):
    """
    Callback function so that our realtime client can fill the audio buffer
    """
    global audio_buffer

    with buffer_lock:
        audio_buffer = np.concatenate([audio_buffer, pcm_audio_chunk])


# callback function for real-time playback using sounddevice
def sd_audio_cb(outdata, frames, time, status):
    global audio_buffer

    channels = 1

    with buffer_lock:
        # if there is enough audio in the buffer, send it
        if len(audio_buffer) >= frames:
            outdata[:] = audio_buffer[:frames].reshape(-1, channels)
            # remove the audio that has been played
            audio_buffer = audio_buffer[frames:]
        else:
            # if not enough audio, fill with silence
            outdata.fill(0)

# Classes
class SimpleRealtime:
    def __init__(self, event_loop=None, audio_buffer_cb=None, debug=False):
        self.url = 'wss://api.openai.com/v1/realtime'
        self.debug = debug
        self.event_loop = event_loop
        self.logs = []
        self.transcript = ""
        self.ws = None
        self._message_handler_task = None
        self.audio_buffer_cb = audio_buffer_cb


    def is_connected(self):
        return self.ws is not None and self.ws.open


    def log_event(self, event_type, event):
        if self.debug:
            local_timezone = tzlocal.get_localzone() 
            now = datetime.now(local_timezone).strftime("%H:%M:%S")
            msg = json.dumps(event)
            self.logs.append((now, event_type, msg))

        return True

    async def connect(self, model="gpt-4o-realtime-preview-2024-10-01"):
        if self.is_connected():
            raise Exception("Already connected")

        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        self.ws = await websockets.connect(f"{self.url}?model={model}", extra_headers=headers)
        
        # Start the message handler in the same loop as the websocket
        self._message_handler_task = self.event_loop.create_task(self._message_handler())
        
        return True


    async def _message_handler(self):
        try:
            while True:
                if not self.ws:
                    await asyncio.sleep(0.05)
                    continue
                    
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=0.05)
                    data = json.loads(message)
                    self.receive(data)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
        except Exception as e:
            print(f"Message handler error: {e}")
            await self.disconnect()


    async def disconnect(self):
        if self.ws:
            await self.ws.close()
            self.ws = None
        if self._message_handler_task:
            self._message_handler_task.cancel()
            try:
                await self._message_handler_task
            except asyncio.CancelledError:
                pass
        self._message_handler_task = None
        return True


    def handle_audio(self, event):
        if event.get("type") == "response.audio_transcript.delta":
            self.transcript += event.get("delta")

        if event.get("type") == "response.audio.delta" and self.audio_buffer_cb:
            b64_audio_chunk = event.get("delta")
            decoded_audio_chunk = base64.b64decode(b64_audio_chunk)
            pcm_audio_chunk = np.frombuffer(decoded_audio_chunk, dtype=np.int16)
            self.audio_buffer_cb(pcm_audio_chunk)


    def receive(self, event):
        self.log_event("server", event)
        if "response.audio" in event.get("type"):
            self.handle_audio(event)
        return True


    def send(self, event_name, data=None):
        if not self.is_connected():
            raise Exception("RealtimeAPI is not connected")
        
        data = data or {}
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary")
        
        event = {
            "type": event_name,
            **data
        }
        
        self.log_event("client", event)
        
        self.event_loop.create_task(self.ws.send(json.dumps(event)))

        return True


class StreamingAudioRecorder:
    """
    Thanks Sonnet 3.5...
    """
    def __init__(self, sample_rate=24_000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_thread = None


    def callback(self, indata, frames, time, status):
        """
        This will be called for each audio block
        that gets recorded.
        """
        self.audio_queue.put(indata.copy())


    def start_recording(self):
        self.is_recording = True
        self.audio_thread = sd.InputStream(
            dtype="int16",
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.callback,
            blocksize=2_000
        )
        self.audio_thread.start()


    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.audio_thread.stop()
            self.audio_thread.close()


    def get_audio_chunk(self):
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None


def start_audio_stream():
    with sd.OutputStream(callback=sd_audio_cb, dtype="int16", samplerate=24_000, channels=1, blocksize=2_000):
        # keep stream open indefinitely, simulate long duration
        sd.sleep(int(10e6))


@st.cache_resource(show_spinner=False)
def create_loop():
    """
    Creates an event loop we can globally cache and then run in a
    separate thread.  Many, many thanks to
    https://handmadesoftware.medium.com/streamlit-asyncio-and-mongodb-f85f77aea825
    for this tip.  NOTE: globally cached resources are shared across all users
    and sessions, so this is only okay for a local R&D app like this.
    """
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    return loop, thread

st.session_state.event_loop, worker_thread = create_loop()


def run_async(coroutine):
    """
    Helper for running an async function in the globally cached event loop we
    just created.
    """
    return run_coroutine_threadsafe(coroutine, st.session_state.event_loop).result()


@st.cache_resource(show_spinner=False)
def setup_client():
    """
    Globally cached SimpleRealtime client.
    """
    if client := st.session_state.get("client"):
        return client
    return SimpleRealtime(event_loop=st.session_state.event_loop, audio_buffer_cb=audio_buffer_cb, debug=True)

st.session_state.client = setup_client()


if "recorder" not in st.session_state:
       st.session_state.recorder = StreamingAudioRecorder()
if "recording" not in st.session_state:
       st.session_state.recording = False


def toggle_recording():
    st.session_state.recording = not st.session_state.recording

    if st.session_state.recording:
        st.session_state.recorder.start_recording()
    else:
        st.session_state.recorder.stop_recording()
        st.session_state.client.send("input_audio_buffer.commit")
        st.session_state.client.send("response.create")


@st.fragment(run_every=1)
def logs_text_area():
    logs = st.session_state.client.logs

    if st.session_state.show_full_events:
        for _, _, log in logs:
            st.json(log, expanded=False)
    else: 
        for time, event_type, log in logs:
            if event_type == "server":
                st.write(f"{time}\t:green[↓ server] {json.loads(log)['type']}")
            else:
                st.write(f"{time}\t:blue[↑ client] {json.loads(log)['type']}")
    st.components.v1.html(AUTOSCROLL_SCRIPT, height=0)


@st.fragment(run_every=1)
def response_area():
    st.markdown("**conversation**")
    st.write(st.session_state.client.transcript)


@st.fragment(run_every=1)
def audio_player():
    if not st.session_state.audio_stream_started:
        st.session_state.audio_stream_started = True
        start_audio_stream()


@st.fragment(run_every=1)
def audio_recorder():
    if st.session_state.recording:
        # drain what's in the queue and send it to openai
        while not st.session_state.recorder.audio_queue.empty():
            chunk = st.session_state.recorder.audio_queue.get()
            st.session_state.client.send("input_audio_buffer.append", {"audio": base64.b64encode(chunk).decode()})


def st_app():
    """
    Our main streamlit app function.
    """
    st.markdown(HIDE_STREAMLIT_RUNNING_MAN_SCRIPT, unsafe_allow_html=True)

    main_tab, docs_tab = st.tabs(["Console", "Docs"])

    with main_tab:
        st.markdown(f"<img src='{OAI_LOGO_URL}' width='30px'/>   **realtime console**", unsafe_allow_html=True)

        with st.sidebar:
            if st.button("Connect", type="primary"):
                with st.spinner("Connecting..."):
                    try:
                        run_async(st.session_state.client.connect())
                        if st.session_state.client.is_connected():
                            st.success("Connected to OpenAI Realtime API")
                        else:
                            st.error("Failed to connect to OpenAI Realtime API")
                    except Exception as e:
                        st.error(f"Error connecting to OpenAI Realtime API: {str(e)}")

        st.session_state.show_full_events = st.checkbox("Show Full Event Payloads", value=False)
        with st.container(height=300, key="logs_container"):
            logs_text_area()

        with st.container(height=300, key="response_container"):
            response_area()

        button_text = "Stop Recording" if st.session_state.recording else "Send Audio"
        st.button(button_text, on_click=toggle_recording, type="primary")

        _ = st.text_area("Enter your message:", key = "input_text_area", height=200)
        def clear_input_cb():
            """
            Callback that will clear our message input box after the user
            clicks the send button.
            """
            st.session_state.last_input = st.session_state.input_text_area
            st.session_state.input_text_area = ""

        if st.button("Send", on_click=clear_input_cb, type="primary"):
            if st.session_state.get("last_input"):
                try:
                    event = json.loads(st.session_state.get("last_input"))
                    with st.spinner("Sending message..."):
                        event_type = event.pop("type")
                        st.session_state.client.send(event_type, event)
                    st.success("Message sent successfully")
                except json.JSONDecodeError:
                    st.error("Invalid JSON input. Please check your message format.")
                except Exception as e:
                    st.error(f"Error sending message: {str(e)}")
            else:
                st.warning("Please enter a message before sending.")

    with docs_tab:
        st.markdown(DOCS)

    audio_player()

    audio_recorder()


if __name__ == '__main__':
    st_app()