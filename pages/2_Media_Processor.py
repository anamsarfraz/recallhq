import asyncio
import streamlit as st

from constants import KNOWLEDGE_BASE_PATH
from recall_utils import update_state
from rags.text_rag import save_processed_document, generate_tags_and_images
from video_processing.ingest_video import process_uploaded_media, Video


def provide_post_process_info(media_label, media_paths):
    file_content = {'media_label': f"{media_label}", 'content': media_paths}
    print(f'file_content: {file_content}')
    st.success("Media uploaded successfully!")
    st.info("You can now go to the knowledge base and ask questions about the media.")

def update_knowledge_base(media_label, media_paths):
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = {}
    if "indexes" not in st.session_state:
        st.session_state.indexes = {}
    st.session_state.knowledge_base.setdefault(media_label, {})
    st.session_state.indexes.setdefault(media_label, {})
    for media_type, paths in media_paths.items():
        st.session_state.knowledge_base[media_label].setdefault(media_type, []).extend(paths)
    save_processed_document(media_label, media_paths["text_paths"], st.session_state.indexes)
    tags_and_imgs = generate_tags_and_images(media_label, st.session_state.indexes)
    st.session_state.knowledge_base[media_label]["tags"] = tags_and_imgs["tags"]
    st.session_state.knowledge_base[media_label]["title_image"] = tags_and_imgs["title_image"]
    update_state(KNOWLEDGE_BASE_PATH, st.session_state.knowledge_base)

def process_content(is_youtube_link, media_label, content):
    if is_youtube_link:
        youtube_links = content.split(',')
        video_paths = []
        audio_paths = []
        text_paths = []
        for youtube_link in youtube_links:
            video = Video.from_url(youtube_link.strip())
            video.download()
            video_path, audio_path, text_path = video.process_video()
            video_paths.append(video_path)
            audio_paths.append(audio_path)
            text_paths.append(text_path)
    else:
        video_path, audio_path, text_path = process_uploaded_media(content)
        if video_path is None and audio_path is None and text_path is None:
            st.error("Failed to process the uploaded media. Please make sure the media is in a supported format.")
            return
        video_paths = [video_path] if video_path else []
        audio_paths = [audio_path] if audio_path else []
        text_paths = [text_path]

    media_paths = {
        "text_paths": text_paths
    }
    if audio_paths != video_paths:
        media_paths["video_paths"] = video_paths
    if text_paths != audio_paths:
        media_paths["audio_paths"] = audio_paths
    provide_post_process_info(media_label, media_paths)
    update_knowledge_base(media_label, media_paths)

def setup_media_processor_page():
    app_header = st.container()
    file_handler = st.form(key='file_handler')

    with app_header:
        st.title("📝 Media Processor ")
        st.markdown("##### Extract text from video and audio files")
        
    with file_handler:
        media_label = st.text_input(label="Media Tag", placeholder="Enter a required label or tag to identify the media")
        youtube_links = st.text_input(label="🔗 YouTube Link(s)",
                                                    placeholder="Enter your YouTube link(s) to download the video and extract the audio")
        uploaded_media = st.file_uploader("📁 Upload your file", type=['mp4', 'wav', 'txt'])
        submit_button = st.form_submit_button(label="Process Media")

        if media_label and submit_button and (youtube_links or uploaded_media):
            if youtube_links:
                print(f'media_label: {media_label}')
                print(f'youtube_links: {youtube_links}')
                with st.spinner("🔍 Extracting transcript...(might take a while)"):
                    process_content(is_youtube_link=True, media_label=media_label, content=youtube_links)
            if uploaded_media:
                with st.spinner("🔍 Reading the media... (might take a while)"):
                    process_content(is_youtube_link=False, media_label=media_label, content=uploaded_media)
setup_media_processor_page()
