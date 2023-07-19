import cv2
import os
from tqdm import tqdm
def calculate_resized_resolution(original_width, original_height, target_height):
    target_width = int((original_width / original_height) * target_height)
    return target_width, target_height

def split_video(video_path, segment_duration, output_directory):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    original_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_height = 854
    target_width = 480
    segment_frame_count = int(segment_duration * fps)
    num_segments = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) // segment_frame_count
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    txt_file_path = os.path.join(output_directory, "video_paths.txt")
    with open(txt_file_path, "w") as txt_file:
        for i in tqdm(range(num_segments)):
            start_frame = i * segment_frame_count
            end_frame = (i + 1) * segment_frame_count
            segment_file_path = os.path.join(output_directory, f"cctv_{i:05d}.mp4")
            out = cv2.VideoWriter(segment_file_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_width, target_height))
            
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_idx in range(start_frame, end_frame):
                ret, frame = video_capture.read()
                if not ret:
                    break
                resized_frame = cv2.resize(frame, (target_width, target_height))
                out.write(resized_frame)
            
            out.release()
            
            relative_path = os.path.relpath(segment_file_path, output_directory)
            txt_file.write(relative_path + "\n")
    
    video_capture.release()


# 使用示例
video_path = "/workdir/video-retalking/examples/face_pro/cctv_pro.mp4"
segment_duration = 5  # in seconds
output_directory = "/workdir/TalkLip/finetune_cctv/"  # 文件保存路径

split_video(video_path, segment_duration, output_directory)