import json
import os
import random
import string

from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip


# Function to load state
def load_state(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        return {}

# Function to update state
def update_state(file_path, new_state):
    current_state = {}
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            current_state = json.load(f)
    
    current_state.update(new_state)
    
    with open(file_path, "w") as f:
        json.dump(current_state, f)

def generate_random_string(length):
    # Generate a random string of specified length using letters and digits
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def generate_videoclips(new_video_path, video_data, concat=False):
    """Clip each video and its associated audio and then concatenate clips if required
    """
    clips = []
    clip_paths = []

    for v in video_data:
        clip = VideoFileClip(v['video_file']).subclip(*v['timestamps'])
        audio = AudioFileClip(v['video_file']).subclip(*v['timestamps'])
        clip.audio = audio
        clips.append(clip)
    if concat:
        final_clip = concatenate_videoclips(clips)
        rnd_str = generate_random_string(10)
        final_path = os.path.join(new_video_path, rnd_str+'.mp4')
        final_clip.write_videofile(final_path, audio_codec='aac', codec='libx264')
        clip_paths.append(final_path)
        return [final_clip], clip_paths
    else:
        for clip in clips:
            rnd_str = generate_random_string(10)
            clip_out_path = os.path.join(new_video_path, rnd_str+'.mp4')
            clip.write_videofile(clip_out_path, audio_codec='aac', codec='libx264')
            clip_paths.append(clip_out_path)
        return clips, clip_paths
