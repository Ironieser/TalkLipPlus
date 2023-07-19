import os
import subprocess
from tqdm import tqdm

def combine_video_audio(video_folder, audio_folder, ckpt_path, avhubert_root, save_path):
    # 获取视频文件夹中的所有文件
    video_files = [file for file in os.listdir(video_folder) if file.endswith(".mp4")]

    # 获取音频文件夹中的所有文件
    audio_files = [file for file in os.listdir(audio_folder) if file.endswith(".wav")]
    total_combinations = len(video_files) * len(audio_files)
    with tqdm(total=total_combinations, desc='Combining', unit='combination') as pbar:
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)  # 视频文件路径

            for audio_file in audio_files:
                audio_path = os.path.join(audio_folder, audio_file)  # 音频文件路径

                # 组合视频和音频对的命令
                command = f"python inf_demo.py --video_path {video_path} --wav_path {audio_path} --ckpt_path {ckpt_path} --avhubert_root {avhubert_root} --save_path {save_path}"

                # 执行命令
                pbar.set_postfix({'Video': video_file, 'Audio': audio_file})
                # subprocess.run(command, shell=True)
                process = subprocess.Popen(command, shell=True)
                process.wait()
                pbar.update(1) 
# 示例用法
video_folder = "/workdir/video-retalking/examples/face_pro"
audio_folder = "/workdir/video-retalking/examples/audio_pro2"
ckpt_path = "/workdir/TalkLip/ckpt/global_contrastive.pth"
avhubert_root = "/workdir/TalkLip/av_hubert"
save_path = "./new_results_0615"

combine_video_audio(video_folder, audio_folder, ckpt_path, avhubert_root, save_path)
