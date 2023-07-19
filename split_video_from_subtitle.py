import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def split_video_by_subtitles(video_path, subtitles_path, output_directory,data_path,output_directory_text ):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(subtitles_path, "r") as f:
        lines = f.readlines()

    segment_num = 1
    filelist = []

    for i in range(0, len(lines), 4):
        start_time,end_time = lines[i+1].strip().split(" --> ")
        subtitle_text = lines[i+2].strip()

        start_time_sec = convert_time_to_seconds(start_time)
        end_time_sec = convert_time_to_seconds(end_time)

        segment_file_path = os.path.join(output_directory, f"cctv_{segment_num:05d}.mp4")
        ffmpeg_extract_subclip(video_path, start_time_sec, end_time_sec, targetname=segment_file_path)

        filelist.append(segment_file_path)

        txt_file_path = os.path.join(output_directory_text, f"cctv_{segment_num:05d}.txt")
        with open(txt_file_path, "w") as txt_file:
            txt_file.write(f"{segment_file_path}  {subtitle_text}")

        segment_num += 1

    filelist_path = os.path.join(data_path, "filelist.txt")
    with open(filelist_path, "w") as filelist_file:
        for file_name in filelist:
            filelist_file.write(file_name + "\n")

# def convert_time_to_seconds(time_str):
#     h, m, s = map(int, time_str.split(":"))
#     seconds = h * 3600 + m * 60 + s
#     return seconds
def convert_time_to_seconds(time_str):
    h, m, s = map(int, time_str[:-4].split(":"))
    ms = int(time_str[-3:])
    seconds = h * 3600 + m * 60 + s + ms / 1000.0
    return seconds

# 使用示例
data_path = "/workdir/TalkLip/mydata/"
video_path = "/workdir/TalkLip/mydata/cctv_pro.mp4"

subtitles_path = "/workdir/TalkLip/mydata/cctv_pro2.srt"
output_directory = "/workdir/TalkLip/mydata/ori_video"  # 文件保存路径
output_directory_text = "/workdir/TalkLip/mydata/text"
split_video_by_subtitles(video_path, subtitles_path, output_directory,data_path,output_directory_text )
