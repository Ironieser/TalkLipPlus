# TalkLipPlus

This repo is a new **SoTA** method by adds some features based on the **official**  implementation of 'Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert', CVPR 2023.[Paper](http://arxiv.org/abs/2303.17480)

Author: Ironeiser.


# TalkLipPlus

This repository presents **TalkLipPlus**, an improved **SoTA** method that extends the capabilities of the **official** implementation of ['Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert'](http://arxiv.org/abs/2303.17480), CVPR 2023.

🔗 **Project Homepage:** [TalkLipPlus Website](https://ironieser.github.io/TalkLipPlus/)  

**Author:** Sixun Dong


## 🔥Feauture🔥  

- 🛠️ Add **post-processing** to enhance the final performance.
- 📊 Add **data pre-processing** to increase the train data quality.
- 🧑‍🔬 Adjust the method that helps **fix the generated face** into the original video with higher quality.
- 🏆 The result achieves **SoTA** compared with the original TalkLip and other methods before **2023.08**.
- ⏱️ Train from scratch with a **5-minute** video. And infer with **no limited** 🎧 audio input.

--- 
## 🎥 Qualitative Results

Click [here](https://ironieser.github.io/TalkLipPlus/) to watch the videos.

**Click on the images below to play the corresponding videos.** ⬇️

| Ground Truth (GT) | TalkLip | TalkLipPlus |
|-------------------|---------|-----------|
| [![GT](https://github.com/Ironieser/TalkLipPlus/blob/main/1.jpg?raw=true)](https://ironieser.github.io/TalkLipPlus/CCTV_00007.mp4) | [![TalkLip](https://github.com/Ironieser/TalkLipPlus/blob/main/2.jpg?raw=true)](https://ironieser.github.io/TalkLipPlus/CCTV_00007_CCTV_00007_62000_360p_finetune_1080.mp4) | [![TalkLip++](https://github.com/Ironieser/TalkLipPlus/blob/main/3.jpg?raw=true)](https://ironieser.github.io/TalkLipPlus/output_video_test_cctv_gpen_wo_face_enhance_wo_possion_blending_w_gfpgan.mp4) |


## 📖 **Citation**
If you find this repository useful, please cite **TalkLipPlus** using the following BibTeX:

```bibtex
@misc{talklipplus2024,
  author    = {Ironeiser},
  title     = {TalkLipPlus: Improved TalkLip for Talking Face Generation},
  year      = {2024},
  howpublished = {\url{https://github.com/Ironieser/TalkLipPlus}},
}
```


The following sections haven't been changed, just copied from the official repo.  
If you want to run this repo, please try to run it.
```
python run_finetune_cctv.py # xxx is the data name, refer to code.
python run_info_demo_cctv.py #
```
--- 
## Prerequisite 

1. `pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html`.
2. Install [AV-Hubert](https://github.com/facebookresearch/av_hubert) by following his installation.
3. Install supplementary packages via `pip install -r requirements.txt`
5. Install ffmpeg. We adopt version=4.3.2. Please double check wavforms extracted from mp4 files. Extracted wavforms should not contain prefix of 0. If you use anaconda, you can refer to `conda install -c conda-forge ffmpeg==4.2.3`
6. Download the pre-trained checkpoint of face detector [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) and put it to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8).


## Dataset and pre-processing

1. Download [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) for training and evaluation. Note that we do not use the pretrain set.
2. Download [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) for evaluation.
3. To extract wavforms from mp4 files:
```
python preparation/audio_extract.py --filelist $filelist  --video_root $video_root --audio_root $audio_root
```
- $filelist: a txt file containing names of videos. We provide the filelist of LRW test set as an example in the datalist directory.  
- $video_root: root directory of videos. In LRS2 dataset, $video_root should contains directories like "639XXX". In LRW dataset, $video_root should contains directories like "ABOUT". 
- $audio_root: root directory for saving wavforms
- other optional arguments: please refer to audio_extract.py  

4. To detect bounding boxes in videos and save it:
```
python preparation/bbx_extract.py --filelist $filelist  --video_root $video_root --bbx_root $bbx_root --gpu $gpu
```
- $bbx_root: a root directory for saving detected bounding boxes
- $gpu: run bbx_extract on a specific gpu. For example, 3.

  *If you want to accelerate bbx_extract via multi-thread processing, you can use the following bash script:
  
  *Please revise variables in the 2-nd to the 9-th lines to make it compatible with your own machine.
```
sh preprocess.sh
```

- $file_list_dir: a directory which contains train.txt, valid.txt, test.txt of LRS2 dataset
- $num_thread: number of threads you used. Please do not let it cross 8 with a 24GB GPU, 4 with a 12GB gpu.


Checkpoints
----------
| Model  | Description |  Link  | 
| :-------------: | :---------------: | :---------------: |
| TalkLip (g)  | TalkLip net with the global audio encoder | [Link](https://drive.google.com/file/d/1iBXJmkS8rjzTBE6XOC3-XiXufEK2f1dj/view?usp=share_link)  |
| TalkLip (g+c)  | TalkLip net with the global audio encoder and contrastive learning | [Link](https://drive.google.com/file/d/1nfPHicsHr2bOzvkdyoMk_GCYzJ3fqvI-/view?usp=share_link) |
| Lip reading observer 1 | AV-hubert (large) fine-tuned on LRS2 | [Link](https://drive.google.com/file/d/1wOsiXKLOeScrU6XuzebYA6Y-9ncd8-le/view?usp=share_link) |
| Lip reading observer 2 | Conformer lip-reading network | [Link](https://drive.google.com/file/d/16tpyaXLLTYUnIBT_YEWQ5ui6xUkBGcpM/view?usp=share_link) |
| Lip reading expert | lip-reading network for training of talking face generation | [Link](https://drive.google.com/file/d/1XAVhWXjd77UHsfna9O8cASHr3iGiQBQU/view?usp=share_link) |


## Train 
```
python train.py --file_dir $file_list_dir --video_root $video_root --audio_root $audio_root \
--bbx_root $bbx_root --word_root $word_root --avhubert_root $avhubert_root --avhubert_path $avhubert_path \
--checkpoint_dir $checkpoint_dir --log_name $log_name --cont_w $cont_w --lip_w $lip_w --perp_w $perp_w \
--gen_checkpoint_path $gen_checkpoint_path --disc_checkpoint_path $disc_checkpoint_path
```
- $file_list_dir: a directory which contains train.txt, valid.txt, test.txt of LRS2 dataset
- $word_root: root directory of text annotation. Normally, it should be equal to $video_root, as LRS2 dataset puts a video file ".mp4" and its corresponding text file ".txt" in the same directory.
- $avhubert_root: path of root of avhubert (should like xxx/av_hubert/avhubert)
- $avhubert_path: download the above Lip reading expert and enter its path
- $checkpoint_dir: a directory to save checkpoint of talklip
- $log_name: name of log file
- $cont_w: weight of contrastive learning loss (default: 1e-3)
- $lip_w: weight of lip reading loss (default: 1e-5)
- $perp_w: weight of perceptual loss (default: 0.07)
- $gen_checkpoint_path(optional): enter the path of a generator checkpoint if you want to resume training from a checkpoint
- $disc_checkpoint_path(optional): enter the path of a discriminator checkpoint if you want to resume training from a checkpoint

Note: Sometimes, discriminator losses may diverge during training (close to 100). Please stop the training and resume it with a reliable checkpoint.


## Test 
The below command is to synthesize videos for quantitative evaluation in our paper.
```
python inf_test.py --filelist $filelist --video_root $video_root --audio_root $audio_root \
--bbx_root $bbx_root --save_root $syn_video_root --ckpt_path $talklip_ckpt --avhubert_root $avhubert_root
```
- $syn_video_root: root directory for saving synthesized videos
- $talklip_ckpt: a trained checkpoint of TalkLip net


## Demo
I update the inf_demo.py on 4/April as I previously suppose that the height and width of output videos are the same when I set cv2.VideoWriter().
Please ensure the sampling rate of the input audio file is 16000 hz.

If you want to reenact the lip movement of a video with a different speech, you can use the following command. 
```
python inf_demo.py --video_path $video_file --wav_path $audio_file --ckpt_path $talklip_ckpt --avhubert_root $avhubert_root
```
- $video_file: a video file (end with .mp4)
- $audio_file: a audio file (end with .wav)

## Evaluation

Please follow README.md in the evaluation directory


