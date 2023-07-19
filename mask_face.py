from os import path

import numpy as np
import argparse, os, cv2
from tqdm import tqdm
import math

import sys
sys.path.append(os.getcwd().replace('preparation', ''))
import face_detection

def images_to_video(image_folder, output_path, fps):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    image_files.sort()  # 对文件进行排序

    # 从第一张图像中获取图像尺寸作为视频尺寸
    image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(image_path)
    height, width, _ = first_image.shape

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可根据需要更改编码器
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 逐帧写入图像到视频
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video.write(image)

    # 释放资源
    video.release()

    print(f'视频已保存至：{output_path}')
    return output_path

def process_video_file(samplename, video_root, out_root, batch_size, fa):
    vfile = '{}/{}'.format(video_root, samplename)
    output_path= '{}/{}'.format(out_root, samplename)

    video_stream = cv2.VideoCapture(vfile)
    fps = video_stream.get(5)
    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可根据需要更改编码器
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    for fb in (batches):
        preds = fa.get_detections_for_batch(np.asarray(fb))
        for pred, of in zip(preds,fb):
            if pred is not None:
                for bbox in pred:
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        face_region = of[y1:y2,x1:x2,:]
                        face_region_mosaic = cv2.medianBlur(face_region,35)
                        of[y1:y2, x1:x2] = face_region_mosaic
                    # print('save:',number)
                    # cv2.imwrite('frame'+str(number)+'.jpg',of[y1:y2, x1:x2])
            video.write(of)
    video.release()


def main(args, fa):
    print('Started processing of {}-th rank for {} on {} GPUs'.format(args.rank, args.video_root, args.gpu))

    # with open(args.filelist) as f:
    #     lines = f.readlines()

    # filelist = [line.strip().split()[0] for line in lines]

    # nlength = math.ceil(len(filelist) / args.nshard)
    # start_id, end_id = nlength * args.rank, nlength * (args.rank + 1)
    # filelist = filelist[start_id: end_id]
    # print('process {}-{}'.format(start_id, end_id))

    # process_video_file(samplename='001', ) 
    file_list = os.listdir('/workdir/face_data/datasetB')
    file_list.sort()
    pbar = tqdm(file_list)
    number = 0
    for vfile in pbar:
        # vfile = r'stu14_29.mp4'
        pbar.set_description('Number:{}, File:{}'.format(number, vfile))
        process_video_file(samplename = vfile, video_root='/workdir/face_data/datasetB',out_root='/workdir/face_data/mask_datasetB',batch_size=8, fa=fa)
        number+=1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
    parser.add_argument('--filelist', help="Path of a file list containing all samples' name", required=False, type=str)
    parser.add_argument("--video_root", help="Root folder of video", required=False, type=str)
    parser.add_argument('--bbx_root', help="Root folder of bounding boxes of faces", required=False, type=str)
    parser.add_argument("--rank", help="the rank of the current thread in the preprocessing ", default=1, type=int)
    parser.add_argument("--nshard", help="How many threads are used in the preprocessing ", default=1, type=int)
    parser.add_argument("--gpu", help="the rank of the current thread in the preprocessing ", default=1, type=int)

    args = parser.parse_args()

    if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
        raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
    							before running this script!')

    args.rank -= 1
    fa = face_detection.FaceAlignment2(face_detection.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(0))

    main(args, fa)



# if __name__ == '__main__':
#     fa = face_detection.FaceAlignment2(face_detection.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(0))


#     img1 = cv2.imread('/workdir/face_data/Snipaste_2023-07-13_10-34-22.png')
#     img2 = cv2.imread('/workdir/face_data/white.png')
#     img2 = cv2.resize(img2,img1.shape[1::-1])
#     imgs = np.concatenate((img1[np.newaxis,...],img2[np.newaxis,...]),axis=0)
#     preds  = fa.get_detections_for_batch(np.asarray(imgs))
#     print( preds )
#     # process_video_file(samplename='001', video_root='/workdir/face_data/datasetB',out_root='/workdir/face_data/mask_datasetB',batch_size=32, fa=fa)