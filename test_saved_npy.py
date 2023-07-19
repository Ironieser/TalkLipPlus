import numpy as np
import cv2

# 读取.npy文件
data = np.load('/workdir/TalkLip/mydata/cctv_1080/head/CCTV_1080_00000.npy')

# 获取帧数和每帧的尺寸
num_frames, height, width, channels = data.shape

# 定义视频保存路径和编解码器
output_path = '/workdir/TalkLip/mydata/cctv_1080/test_npy_360p.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可根据需要选择合适的编解码器

# 创建视频写入器
video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

# 逐帧写入视频
for frame_index in range(num_frames):
    frame = data[frame_index]

    # 将BGR格式转换为RGB格式
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 将RGB帧写入视频
    video_writer.write(frame)

# 释放资源
video_writer.release()