import cv2
import numpy as np
# 假设视频文件为video.mp4
# video_file = "/workdir/TalkLip/finetune_cctv/video/cctv_00000.mp4"
video_file =        "/workdir/TalkLip/mydata/cctv/1080pp/video/CCTV_00000.mp4"

# 假设坐标数据的npy文件为coords.npy
coords_file =       "/workdir/TalkLip/mydata/cctv/1080pp/bbx/CCTV_00000.npy"
coords_file_down =  "/workdir/TalkLip/mydata/cctv/test/bbx/CCTV_00000_ori.npy"
head_file =         "/workdir/TalkLip/mydata/cctv/1080pp/head/CCTV_00000.npy"

# 指定调整后的尺寸
target_size = (96, 96)

# 读取视频
cap = cv2.VideoCapture(video_file)

# 读取坐标数据
coords = np.load(coords_file)
coords_down = np.load(coords_file_down)
head = np.load(head_file)

# 获取视频的原始尺寸和帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 创建输出视频的写入器
# output_file = "./test_resize/test_96_cctv_1080ppin360p_round_avg.mp4"
output_file = "./test_resize/test_96_cctv_1080ppin1080pp_newbbx_ori_lin.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# 遍历每一帧并处理人头坐标
frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 获取当前帧的人头坐标
    if frame_index < len(coords):
        head_coords = coords[frame_index]
        head_coords_down = coords_down[frame_index]
        x1, y1, x2, y2 = head_coords_down

        # x1, y1, x2, y2 = np.round(head_coords/2,decimals=0).astype(int)
        
        # avg_head_coords = head_coords/3 + head_coords_down 
        # x1, y1, x2, y2 = np.round(avg_head_coords/2,decimals=0).astype(int)
        
        # resized_head = cv2.resize(head[frame_index], (x2 - x1,y2 - y1 ),interpolation=cv2.INTER_CUBIC)
        resized_head = cv2.resize(head[frame_index], (x2 - x1,y2 - y1 ),)
        # 将人头区域尺寸调整回原始尺寸
        # resized_head = cv2.resize(resized_head, (x2 - x1, y2 - y1))

        # 将调整后的人头区域放回原始帧
        frame[y1:y2, x1:x2] = resized_head
    
    # 写入输出视频
    out.write(frame)
    frame_index += 1

# 释放资源
cap.release()
out.release()