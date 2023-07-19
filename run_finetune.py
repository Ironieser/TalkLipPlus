import subprocess

args = [
    "finetune_wo_txt.py",
    "--file_dir", "/workdir/TalkLip/finetune_data/datalist",
    "--avhubert_root", "/workdir/TalkLip/av_hubert",
    "--avhubert_path", "/workdir/TalkLip/ckpt/lip_reading_expert.pt",
    "--checkpoint_dir", "./finetune_ckpt",
    "--log_name", "./fintune_log",
    "--cont_w", "1e-3",
    "--lip_w", "1e-5",
    "--perp_w", "0.07",
    "--ckpt_interval", "100",
    "--n_epoch", "500",
    "--num_worker", "0",
    "--batch_size", "2",
    "--gen_checkpoint_path", "/workdir/TalkLip/finetune_ckpt/checkpoint_step000061800.pth",
    "--disc_checkpoint_path","/workdir/TalkLip/finetune_ckpt/disc_checkpoint_step000061800.pth",
    "--video_root", "./finetune_data/video",
    "--audio_root", "./finetune_data/norm_audio",
    "--bbx_root", "./finetune_data/bbx"
]

# 构建运行 finetune_wo_txt.py 的命令
command = ["python"] + args

# 执行命令
subprocess.run(command)