
import subprocess

command = [
    "python", 
    "inf_demo.py",
    "--video_path", "/workdir/TalkLip/mydata/cctv/1080pp/video/CCTV_00007.mp4",
    "--wav_path", "/workdir/TalkLip/mydata/cctv/1080pp/norm_audio/CCTV_00007.wav",
    # "--ckpt_path", "/workdir/TalkLip/finetune_cctv_ckpt/checkpoint_step000062100.pth",
    "--ckpt_path", "/workdir/TalkLip/finetune_cctv_head_1080_ckpt/checkpoint_step000062200.pth",
    "--avhubert_root", "/workdir/TalkLip/av_hubert",
    "--save_path", "/workdir/TalkLip/test_results_cctv_0710",
    "--device","0",
    "--out_name","1080p_finetune_1080",
]

# 执行命令p
subprocess.run(command)