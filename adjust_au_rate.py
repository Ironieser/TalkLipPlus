import os
import librosa
import soundfile as sf

def resample_audio_files(input_folder, output_folder, target_sr):
    audio_files = [file for file in os.listdir(input_folder) if file.endswith('.wav')]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for audio_file in audio_files:
        file_path = os.path.join(input_folder, audio_file)

        # 读取音频文件
        audio, sr = librosa.load(file_path, sr=None)

        # 调整采样率
        resampled_audio = librosa.resample(y = audio, orig_sr = sr, target_sr = target_sr)

        # 保存音频文件
        output_file_path = os.path.join(output_folder, audio_file)
        sf.write(output_file_path, resampled_audio, target_sr, format='wav')

# 调用示例
input_folder = '/workdir/video-retalking/examples/norm_audio'  # 输入文件夹路径
output_folder = '/workdir/video-retalking/examples/norm_audio'  # 输出文件夹路径
target_sr = 16000  # 目标采样率

resample_audio_files(input_folder, output_folder, target_sr)