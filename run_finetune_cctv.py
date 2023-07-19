import subprocess

args = [
    "finetune_wo_txt.py",
    "--file_dir", "/workdir/TalkLip/finetune_cctv/datalist",
    "--avhubert_root", "/workdir/TalkLip/av_hubert",
    "--avhubert_path", "/workdir/TalkLip/ckpt/lip_reading_expert.pt",
    "--checkpoint_dir", "/workdir/TalkLip/finetune_cctv_ckpt",
    "--log_name", "train_cctv_log2",
    "--cont_w", "1e-3",
    "--lip_w", "1e-5",
    "--perp_w", "0.05",
    "--ckpt_interval", "50",
    "--n_epoch", "300",
    "--num_worker", "0",
    "--batch_size", "2",
    "--gen_checkpoint_path", "/workdir/TalkLip/finetune_cctv_ckpt/checkpoint_step000061200.pth",
    "--disc_checkpoint_path", "/workdir/TalkLip/finetune_cctv_ckpt/disc_checkpoint_step000061200.pth",
    "--video_root", "/workdir/TalkLip/finetune_cctv/video",
    "--audio_root", "/workdir/TalkLip/finetune_cctv/norm_audio",
    "--bbx_root", "/workdir/TalkLip/finetune_cctv/bbx"
]

# 构建运行 finetune_wo_txt.py 的命令
command = ["python"] + args

# 执行命令
subprocess.run(command)