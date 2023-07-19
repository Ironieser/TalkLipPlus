import subprocess

args = [
    "finetune_wo_txt_change_gan.py",
    "--avhubert_root", "/workdir/TalkLip/av_hubert",
    "--avhubert_path", "/workdir/TalkLip/ckpt/lip_reading_expert.pt",
    "--checkpoint_dir", "/workdir/TalkLip/finetune_cctv_head_1080_ckpt",
    "--log_name", "finetune_cctv_head_1080",
    "--cont_w", "1e-3",
    "--lip_w", "1e-5",
    "--perp_w", "0.05",
    "--ckpt_interval", "100",
    "--n_epoch", "5000",
    "--num_worker", "0",
    "--batch_size", "2",
    "--gen_checkpoint_path", "/workdir/TalkLip/finetune_cctv_head_1080_ckpt/checkpoint_step000062000.pth",
    # "--gen_checkpoint_path", "/workdir/TalkLip/ckpt/global_contrastive.pth",
    "--disc_checkpoint_path", "/workdir/TalkLip/finetune_cctv_head_1080_ckpt/disc_checkpoint_step000062000.pth",
    "--file_dir", "/workdir/TalkLip/mydata/cctv/1080pp/datalist",
    "--video_root", "/workdir/TalkLip/mydata/cctv/1080pp/video",
    "--audio_root", "/workdir/TalkLip/mydata/cctv/1080pp/norm_audio",
    "--bbx_root", "/workdir/TalkLip/mydata/cctv/1080pp/bbx"
]

# 构建运行 finetune_wo_txt.py 的命令
command = ["python"] + args

# 执行命令
subprocess.run(command)