import os
import numpy as np
import cv2

# 定义文件列表的路径和视频文件夹路径
filelist_path = '/workdir/TalkLip/mydata/cctv_srt/filelist.txt'
video_folder = '/workdir/TalkLip/mydata/cctv_srt/ori_video'
bbx_folder = '/workdir/TalkLip/mydata/cctv_srt/bbx'
head_folder = '/workdir/TalkLip/mydata/cctv_srt/head'
target_size = (96, 96)

# 创建保存头部数据的文件夹
if not os.path.exists(head_folder):
    os.makedirs(head_folder)

# 读取文件列表
with open(filelist_path, 'r') as file:
    filenames = file.read().splitlines()

# 遍历文件列表
erro = 0
for filename in filenames:
    # 处理视频文件
    video_path = os.path.join(video_folder, filename + '.mp4')
    if os.path.exists(video_path):
        # 处理头部坐标文件
        bbx_path = os.path.join(bbx_folder, filename + '.npy')

        if os.path.exists(bbx_path):
            # 读取头部坐标数据
            cap = cv2.VideoCapture(video_path)
            coords = np.load(bbx_path)
            frame_index = 0
            result_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 获取当前帧的人头坐标
                if frame_index < len(coords):
                    head_coords = coords[frame_index]
                    x1, y1, x2, y2 = head_coords
                    width = frame.shape[1]
                    if x2>width or y2>width:
                        print(frame_index)
                        erro +=1
                    # 调整人头区域尺寸
                    resized_head = cv2.resize(frame[y1:y2, x1:x2], target_size,interpolation=cv2.INTER_CUBIC)
                    result_frames.append(resized_head)

                frame_index += 1
            cap.release()
            head_data = np.stack(result_frames, axis=0)
            # 保存头部数据为npy格式
            head_save_path = os.path.join(head_folder, filename + '.npy')
            np.save(head_save_path, head_data)
        else:
            print(f"找不到文件: {bbx_path}")
    else:
        print(f"找不到文件: {video_path}")

print('erro:',erro)