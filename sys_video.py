import cv2
import os

from moviepy.editor import VideoFileClip, AudioFileClip

def images_to_video(image_folder, output_path, fps):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    image_files.sort()  # 对文件进行排序

    # 从第一张图像中获取图像尺寸作为视频尺寸
    image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(image_path)
    height, width, _ = first_image.shape

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可根据需要更改编码器
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 逐帧写入图像到视频
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video.write(image)

    # 释放资源
    video.release()

    print(f'视频已保存至：{output_path}')
    return output_path
# 设置输入图像文件夹、输出视频路径和帧率
input_folder = '/workdir/TalkLip/output_img_gfpgan/cctv_4/restored_imgs'  # 输入图像文件夹路径
output_video = '/workdir/TalkLip/gfpgan_results/cctv_3/output.mp4'  # 输出视频路径
frame_rate = 25  # 视频帧率

# 调用函数将图像合成为视频
video_path = images_to_video(input_folder, output_video, frame_rate)


# 合成的视频文件路径
video_path = video_path

# 指定的音频文件路径
# audio_path = '/workdir/TalkLip/mydata/generation/audio/5_20.wav'
audio_path = '/workdir/TalkLip/mydata/cctv/1080pp/norm_audio/CCTV_00007.wav'

# 输出视频文件路径
output_path = '/workdir/TalkLip/gfpgan_results/cctv_3/output_video.mp4'

# 加载合成的视频文件和音频文件
video = VideoFileClip(video_path)
audio = AudioFileClip(audio_path)

# 将音频文件与视频文件进行合并
video_with_audio = video.set_audio(audio)

# 保存合并后的视频文件
video_with_audio.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=video.fps)

# 关闭视频和音频文件对象
video.close()
audio.close()