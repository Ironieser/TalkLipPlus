
import subprocess

command = [
    "python", 
    "inf_demo.py",
    "--video_path", "/workdir/TalkLip/finetune_data/video/MAGGIE_00000.mp4",
    "--wav_path", "/workdir/TalkLip/finetune_data/norm_audio/MAGGIE_00008.wav",
    "--ckpt_path", "/workdir/TalkLip/finetune_ckpt/checkpoint_step000067300.pth",
    "--avhubert_root", "/workdir/TalkLip/av_hubert",
    "--save_path", "./test_results_0615_67300",
    "--device","0",
]

# 执行命令
subprocess.run(command)