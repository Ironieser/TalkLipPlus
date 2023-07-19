import os

folder_path = '/workdir/TalkLip/mydata/maggie_gl/1080p/video'

# 获取目标文件夹中的所有文件名python
file_names = os.listdir(folder_path)

# 遍历文件名
for file_name in file_names:
    # 检查文件名是否匹配所需格式
    if file_name.startswith('maggie_') and file_name.endswith('.mp4'):
        # 构建新的文件名
        new_file_name = 'maggie_gl_' + file_name[-9:-4] + '.mp4'  # 根据您的需求构建新的文件名

        # 构建旧文件的完整路径和新文件的完整路径
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)

        # 使用 os.rename() 函数进行文件名更改
        os.rename(old_file_path, new_file_path)
