
import subprocess

command = [
    "python", 
    "inf_demo.py",
    "--video_path", "/workdir/TalkLip/mydata/maggie_gl/1080p/video/maggie_gl_00000.mp4",
    # "--wav_path", "/workdir/TalkLip/mydata/maggie_gl/1080p/norm_audio/maggie_gl_00024.wav",
    "--wav_path", "/workdir/TalkLip/mydata/generation/audio/4_20.wav",
    "--ckpt_path", "/workdir/TalkLip/finetune_maggie_gl_head_1080_ckpt_2/checkpoint_step000063000.pth",
    # "--ckpt_path", "/workdir/TalkLip/finetune_ckpt/checkpoint_step000063000.pth",
    "--avhubert_root", "/workdir/TalkLip/av_hubert",
    "--save_path", "/workdir/TalkLip/test_results",
    "--device","0",
    "--out_name","1080p_finetune_1080_2",
]

# 执行命令p
subprocess.run(command)