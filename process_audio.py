import os
import librosa
import soundfile as sf

def normalize_audio_files(input_folder, output_folder,target_sr):
    audio_files = [file for file in os.listdir(input_folder) if file.endswith('.wav') or file.endswith('.mp3')]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for audio_file in audio_files:
        file_path = os.path.join(input_folder, audio_file)

        audio, sr = librosa.load(file_path, sr=None)
        max_amplitude = max(abs(audio))
        normalized_audio = audio / max_amplitude
        resampled_audio = librosa.resample(y = normalized_audio, orig_sr = sr, target_sr = target_sr)
        output_file_path = os.path.join(output_folder, audio_file)
        sf.write(output_file_path, resampled_audio, target_sr)


# 调用示例
input_folder = '/workdir/TalkLip/mydata/cctv/720p/audio'  # 输入文件夹路径
output_folder = '/workdir/TalkLip/mydata/cctv/720p/norm_audio'
target_sr = 16000

normalize_audio_files(input_folder, output_folder,target_sr)

