from pytubefix import YouTube
import subprocess
import os


def convert_m4a_to_mp3(input_path, output_path):
    try:
        command = [
            "ffmpeg",
            "-i",
            input_path,
            output_path,
        ]
        subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        print(f"Conversion successful")
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")


def download_audio(url):

    yt = YouTube(url)
    ys = yt.streams.get_audio_only()
    path = ys.download()
    new_path = yt.title + ".mp3"
    convert_m4a_to_mp3(path, new_path)
    os.remove(path)
    return new_path
