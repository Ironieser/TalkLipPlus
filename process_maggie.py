# 要实现视频同名文件已存在时直接覆盖的功能，你可以在调用FFmpeg之前先检查目标文件是否存在，如果存在，则先删除该文件，然后再进行视频切分和保存。以下是修改后的代码示例：

import os
import subprocess
from tqdm import tqdm

def split_video(input_file, output_directory, resolution,txt_path ):
    txt_file = open(os.path.join(txt_path +"video_paths.txt"), "w")
    count = 0

    ffprobe_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
    duration = float(subprocess.check_output(ffprobe_command))

    segment_duration = 10
    start_time = 0
    end_time = segment_duration

    pbar = tqdm(total=int(duration), unit='sec', unit_scale=True)

    while end_time <= duration:
        filename = f"maggie_{str(count).zfill(5)}.mp4"
        output_file = os.path.join(output_directory, filename)

        # 如果目标文件已存在，则先删除
        if os.path.exists(output_file):
            os.remove(output_file)

        ffmpeg_command = ['ffmpeg', '-i', input_file, '-ss', str(start_time), '-t', str(segment_duration), '-s', resolution, '-c:v', 'libx264', '-crf', '23', '-preset', 'ultrafast', '-c:a', 'copy', output_file]
        subprocess.call(ffmpeg_command)

        txt_file.write(f"maggie_{str(count).zfill(5)}\n")

        start_time = end_time
        end_time += segment_duration
        count += 1
        pbar.update(segment_duration)

    txt_file.close()
    pbar.close()

# 调用示例
txt_path = "/workdir/TalkLip/mydata/maggie_gl/1080p"
input_file = "/workdir/TalkLip/mydata/maggie_gl/maggie_gl.mp4"  # 输入视频路径
output_directory = "/workdir/TalkLip/mydata/maggie_gl/1080p/video"  # 输出目录
resolution = "1080x1920"  # 输出视频分辨率
split_video(input_file, output_directory, resolution,txt_path )