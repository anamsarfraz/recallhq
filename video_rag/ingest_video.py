# Process a video by extracting knowledge from it into a rag.

from pytubefix import YouTube
from pytubefix.cli import on_progress
from pathlib import Path
from moviepy.editor import VideoFileClip
import speech_recognition as sr


# SET CONFIG
_video_url = "https://www.youtube.com/watch?v=d_qvLDhkg00"
output_video_folder = "./temp/video_data/"
output_mixed_folder = "./temp/mixed_data/"
output_audio_path = "./temp/mixed_data/output_audio.wav"

filepath = output_video_folder + "input_vid.mp4"


def make_tempdirs():
    Path(output_video_folder).mkdir(parents=True, exist_ok=True)
    Path(output_mixed_folder).mkdir(parents=True, exist_ok=True)


def download_video(url, output_path, output_filename, subtitles_outfile):
    """
    Download a video from a given url and save it to the output path.

    Parameters:
    url (str): The url of the video to download.
    output_path (str): The path to save the video to.

    Returns:
    dict: A dictionary containing the metadata of the video.
    """

    yt = YouTube(url, on_progress_callback=on_progress)
    metadata = {"Author": yt.author, "Title": yt.title, "Views": yt.views}
    yt.streams.get_highest_resolution().download(
        output_path=output_path, filename=output_filename
    )
    if subtitles_outfile is not None:
        print(yt.captions)
        caption = yt.captions.get_by_language_code('en')
        if caption is not None:
            caption.save_captions(subtitles_outfile)
    return metadata

def extract_audio(video_path, output_audio_path):
    """
    Convert a video to audio and save it to the output path.

    Parameters:
    video_path (str): The path to the video file.
    output_audio_path (str): The path to save the audio to.

    """
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)

def extract_text(audio_path, text_outfile):
    """
    Extracts text from an audio file using whisper speech recognition (for English only)
    """
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)

    with audio as source:
        # Record the audio data
        audio_data = recognizer.record(source)

        try:
            # Recognize the speech
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from service; {e}")

    return text

video_outfile = "video_one.mp4"
audio_outfile = "audio_one.wav"
text_outfile = "text_output.txt"

def process(url):
    make_tempdirs()
    #print(download_video(url, output_video_folder, video_outfile, "video_one_subtitles.txt"))
    video_path = f"{output_video_folder}/{video_outfile}"
    audio_path = f"{output_video_folder}/{audio_outfile}"
    text_path = f"{output_video_folder}/{text_outfile}"
    extract_audio(video_path, audio_path)
    text = extract_text(audio_path, text_path)
    # Save text to file
    with open(text_path, 'w') as file:
        file.write(text)


process(_video_url)