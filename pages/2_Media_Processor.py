import asyncio
import streamlit as st
from video_processing.ingest_video import process_video, process_uploaded_media

def provide_post_process_info(media_label, media_paths):
    file_content = {'media_label': f"{media_label}", 'content': media_paths}
    print(f'file_content: {file_content}')
    st.success("Media uploaded successfully!")
    st.info("You can now go to the knowledge base and ask questions about the media.")

def setup_media_processor_page():
    app_header = st.container()
    file_handler = st.form(key='file_handler')

    with app_header:
        st.title("📝 Media Processor ")
        st.markdown("##### Extract text from video and audio files")
        
    with file_handler:
        media_label = st.text_input(label="Media Tag", placeholder="Enter a required label or tag to identify the media")
        youtube_link = st.text_input(label="🔗 YouTube Link",
                                                    placeholder="Enter your YouTube link to download the video and extract the audio")
        uploaded_media = st.file_uploader("📁 Upload your file", type=['mp4', 'wav'])
        submit_button = st.form_submit_button(label="Process Media")

        if media_label and submit_button and (youtube_link or uploaded_media):
            if youtube_link and uploaded_media:
                st.warning("Either enter a YouTube link or upload a file, not both.")
            elif youtube_link:
                print(f'media_label: {media_label}')
                print(f'youtube_link: {youtube_link}')
                with st.spinner("🔍 Extracting transcript..."):
                    video_path, audio_path, text_path = process_video(youtube_link)
                    provide_post_process_info(media_label, [video_path, audio_path, text_path])
            else:
                with st.spinner("🔍 Reading file... (might take a while)"):
                    video_path, audio_path, text_path = process_uploaded_media(uploaded_media)
                    provide_post_process_info(media_label, [video_path, audio_path, text_path])

setup_media_processor_page()
