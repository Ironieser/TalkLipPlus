import cv2
import os

# 视频文件路径
video_path = '/workdir/TalkLip/test_results_cctv/CCTV_00007_CCTV_00007_62200_la_1080p_finetune_1080.mp4'

# 保存帧图像的文件夹路径
output_folder = '/workdir/TalkLip/video2imgs/cctv_2/'

# 创建保存帧图像的文件夹
os.makedirs(output_folder, exist_ok=True)

# 打开视频文件
video = cv2.VideoCapture(video_path)

# 逐帧读取视频，并保存为图像文件
frame_count = 0
while True:
    # 读取一帧图像
    ret, frame = video.read()
    if not ret:
        break

    # 生成文件名
    frame_name = f"video_{str(frame_count+1).zfill(5)}.jpg"

    # 保存图像文件
    frame_path = os.path.join(output_folder, frame_name)
    cv2.imwrite(frame_path, frame)

    frame_count += 1

# 释放视频对象
video.release()