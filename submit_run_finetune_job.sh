#!/bin/bash
#SBATCH --job-name=finetune_talklip        # 任务名称
#SBATCH --gres=gpu:1             # 申请一个 GPU
#SBATCH --cpus-per-task=10       # 申请 10 个 CPU 核心
#SBATCH --time=2-00:00:00          # 最大运行时间
#SBATCH --output=output.log      # 输出日志文件

# 定义变量
docker_image="localhost/sixun-talklip:zsh-cuda11.3-cudnn8-devel-ubuntu20.04" 
container_name="talklip_sbatch1"
host_project_path="/public/homes/sixun.dong/project"
container_workdir="/workdir"

# 启动特定的 Docker 容器
podman run -it \
    -v "$host_project_path":"$container_workdir" \
    -w "$container_workdir" \
    -p 12555:2555 \
    --name="$container_name" \
    "$docker_image" \
    /bin/zsh

# 在容器内激活 conda 环境并执行 run_finetune.py 脚本
podman exec "$container_name" conda run -n talklip python /workdir/TalkLip/run_finetune.py

# 停止并删除 Docker 容器
podman stop "$container_name"
podman rm "$container_name"

# 释放资源
exit