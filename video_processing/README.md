# Video Rag tool

On Linux:

```sh 
sudo apt update && sudo apt install ffmpeg
```

or MacOS:
```sh
brew install ffmpeg
```


Set `video_url` in `ingest_video.py` and:

```sh
 python3 video_rag/ingest_video.py -h
 ```

 ```sh
 python3 video_rag/ingest_video.py --youtube_url "https://www.youtube.com/watch?v=SYRunzR9fbk" --output_folder "./t1"
 ```

 ```sh
 python3 video_rag/ingest_video.py --youtube_url "https://www.youtube.com/watch?v=SYRunzR9fbk" 

 ```
 ```sh
 python3 video_rag/ingest_video.py -y "https://www.youtube.com/watch?v=SYRunzR9fbk" 
 ```
