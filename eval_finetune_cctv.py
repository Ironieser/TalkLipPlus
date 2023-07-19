import os
import subprocess
from datetime import datetime
from tqdm import tqdm

current_date = datetime.now()
date = current_date.strftime("%m%d") + '03'

ckpt_path = "/workdir/TalkLip/scatch_cctv_ckpt"
script_dir = os.path.dirname(os.path.abspath(__file__))


device_idx = 0  # 当前显卡索引
video_path = "/workdir/TalkLip/finetune_cctv/video_backup/cctv_train.mp4"
wav_path = "/workdir/video-retalking/examples/audio_pro2/cctv_50.wav"
avhubert_root = "/workdir/TalkLip/av_hubert"

total_files = sum(file.endswith(".pth") and file.startswith("checkpoint_step") for file in os.listdir(ckpt_path))
with tqdm(total=total_files, desc='Processing') as pbar:
    for file in os.listdir(ckpt_path):
        if file.endswith(".pth") and file.startswith("checkpoint_step"):
            checkpoint_step = file.split(".")[0].split("_")[-1][-5:]
            save_path = f"./scratch__cctv_results_{date}"
            command_str = f"python inf_demo.py --video_path {video_path} --wav_path {wav_path} --ckpt_path {os.path.join(ckpt_path, file)} --avhubert_root {avhubert_root} --save_path {save_path} --device {device_idx} --ckpt_step {checkpoint_step}"

            # 更新进度条描述信息，显示当前正在测试的 checkpoint_step
            pbar.set_description(f"Processing checkpoint_step: {checkpoint_step}")
            pbar.update(1) 
            # 执行命令
            process = subprocess.Popen(command_str, shell=True)
            process.wait()