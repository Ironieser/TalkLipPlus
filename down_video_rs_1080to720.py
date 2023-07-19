import os
import subprocess

# 输入文件夹路径
input_folder = '/workdir/TalkLip/mydata/maggie_gl/1080p/video'
# 输出文件夹路径
output_folder = '/workdir/TalkLip/mydata/maggie_gl/720p/video'

# 获取输入文件夹中的所有视频文件
input_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# 遍历每个视频文件并进行分辨率转换
for input_file in input_files:
    # 构建输入文件路径和输出文件路径
    input_path = os.path.join(input_folder, input_file)
    output_file = os.path.splitext(input_file)[0] + '.mp4'
    output_path = os.path.join(output_folder, output_file)

    # 使用FFmpeg命令进行分辨率转换
    command = f'ffmpeg -i "{input_path}" -vf scale=720:1280 "{output_path}"'
    subprocess.call(command, shell=True)

    print(f'Converted: {input_file} -> {output_file}')

print('Conversion complete!')