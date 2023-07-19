
import subprocess

command = [
    "python", 
    "inf_demo_la_gfpgan.py",
    "--video_path", "/workdir/TalkLip/mydata/maggie_gl/1080p/video/maggie_gl_00000.mp4",
    # "--wav_path", "/workdir/TalkLip/mydata/maggie_gl/1080p/norm_audio/maggie_gl_00000.wav",
    "--wav_path", "/workdir/TalkLip/mydata/generation/audio/4_20.wav",
    # "--ckpt_path", "/workdir/TalkLip/finetune_maggie_gl_head_1080_ckpt_2/checkpoint_step000060100.pth",
    "--ckpt_path", "/workdir/TalkLip/finetune_maggie_gl_head_1080_ckpt_2/checkpoint_step000063000.pth",
    "--avhubert_root", "/workdir/TalkLip/av_hubert",
    "--save_path", "/workdir/TalkLip/test_results_restored2",
    "--device","0",
    "--out_name","la_1080p_finetune_1080",
]

# 执行命令p
subprocess.run(command)