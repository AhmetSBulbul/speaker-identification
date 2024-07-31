import subprocess


def convert2wav(source, dest):
    command = ["ffmpeg", "-i", source, "-vn", dest]
    if subprocess.run(command).returncode == 0:
        print ("FFmpeg Script Ran Successfully")
    else:
        print ("There was an error running your FFmpeg script")