import streamlit as st 
import app_utils as vutil 

st.set_page_config(
    page_title="RecallHQ",
    page_icon="recallhq_icon.svg",
)
st.markdown(
    """<style>
    #MainMenu {visibility: hidden;} /* Hides the entire menu in the top-right corner */
    header {visibility: hidden;} /* Hides the top-right Streamlit header (including the 'Deploy' button) */
       </style>""",
    unsafe_allow_html=True
)

def st_button(url, label, font_awesome_icon):
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">', unsafe_allow_html=True)
    button_code = f'''<a href="{url}" target=_blank><i class="fa {font_awesome_icon}"></i>   {label}</a>'''
    return st.markdown(button_code, unsafe_allow_html=True)

def render_cta():
  with st.sidebar:
      st.write("Let's connect!")
      st_button(url="https://www.linkedin.com/company/recallhq", label="LinkedIn", font_awesome_icon="fa-linkedin")

render_cta()

# T
home_title = "RecallHQ"
home_introduction = "Welcome to RecallHQ, where the power of LLM technology is at your fingertips. Upload a video or Youtube link to a video. Then interact  with pre-trained AI Assistants in the Event Q&A. Whether you need to ask questions about the event, or generate summaries, RecallHQ has you covered. Let's start exploring the endless possibilities!"
home_privacy = "At RecallHQ, your privacy is our top priority. We use OpenAI's Whisper API to extract audio from videos and Youtube links. We do not store any data beyond what is required to process the audio and generate summaries."
getstarted_prompt = "Ready to explore the endless possibilities of AI? Let's get started today!"




#st.title(home_title)
st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Beta</font></span>""",unsafe_allow_html=True)

st.markdown("""\n""")
st.markdown("#### Greetings")
st.write(home_introduction)

st.markdown("#### Privacy")
st.write(home_privacy)

st.markdown("""\n""")
st.markdown("""\n""")

st.markdown("#### Get Started")
st.markdown(getstarted_prompt)
st.markdown("\n")
