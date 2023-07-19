from torchvision import transforms
import torch
import cv2
import random
from torchvision.transforms import Resize,InterpolationMode
import numpy as np
def collater_audio(audios, audio_size):
    audio_feat_shape = list(audios[0].shape[1:])
    collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
    padding_mask = (
        torch.BoolTensor(len(audios), audio_size).fill_(False) #
    )
    for i, audio in enumerate(audios):
        diff = len(audio) - audio_size
        if diff == 0:
            collated_audios[i] = audio
        elif diff < 0:
            collated_audios[i] = torch.cat(
                [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
            )
            padding_mask[i, diff:] = True
        else:
            import sys
            sys.exit('Audio segment is longer than the loggest')
    if len(audios[0].shape) == 2:
        collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
    else:
        collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
    return collated_audios, padding_mask


class Compose(object):  
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


transform = Compose([
    Normalize(0.0, 255.0),
    CenterCrop((88, 88)),
    Normalize(0.421, 0.165)])


def rgb2gray(g, dim):
    glist = g.split([1,1,1], dim=dim)
    return 0.299 * glist[2] + 0.587 * glist[1] + 0.114 * glist[0]


def affine_trans(imgs, video_size):
    h, w, _ = imgs[0][0].shape
    videoSeq = list()
    for i, img in enumerate(imgs):
        new_images = list()
        for j, frame in enumerate(img):
            frame = rgb2gray(frame, 2).squeeze(dim=-1)
            new_images.append(frame)
        new_images = torch.stack(new_images, dim=0)
        videoSeq.append(transform(new_images).unsqueeze(dim=-1))
    collated_videos, padding_mask = collater_audio(videoSeq, video_size)
    return collated_videos

def mask_postprocess(mask, thres=20):
        h,w= mask.shape
        # mask = cv2.resize(mask,(512,512))
        # mask[:thres, :] = 0; mask[-thres:, :] = 0
        # mask[:, :thres] = 0; mask[:, -thres:] = 0        
        mask = cv2.GaussianBlur(mask, (51,51), 11)
        mask = cv2.GaussianBlur(mask, (51,51), 11)
        mask = cv2.GaussianBlur(mask, (51,51), 11)
        mask = cv2.GaussianBlur(mask, (51,51), 11)
        # mask = cv2.GaussianBlur(mask, (51,51), 11)
        # mask = cv2.GaussianBlur(mask, (51,51), 11)
        # mask = cv2.resize(mask,(w,h))
        return mask.astype(np.float32)

def emb_roi2im2(pickedimg, imgs, bbxs, pre, device,):
    trackid = 0
    height, width = imgs[0][0].shape[:2]
    for i in range(len(pickedimg)):
        idimg = pickedimg[i]
        imgs[i] = imgs[i].float().to(device)
        for j in range(len(idimg)):
            bbx = bbxs[i][idimg[j]]
            if bbx[2] > width: bbx[2] = width
            if bbx[3] > width: bbx[3] = width
            resize2ori = transforms.Resize([bbx[3] - bbx[1], bbx[2] - bbx[0]],interpolation=InterpolationMode.BICUBIC)
            try:
                resized = resize2ori(pre[trackid + j] * 255.).permute(1, 2, 0)
                # add mask
                delta1 = int((bbx[3]-bbx[1])/10)
                delta2 = int((bbx[2]-bbx[0])/10)
                full_mask = np.ones((height, width), dtype=np.float32)
                full_mask[bbx[1]+delta1:bbx[3]-delta1, bbx[0]+delta2:bbx[2]-delta2] = 0
                gauss_mask = mask_postprocess(full_mask)
                full_mask = 1 - torch.from_numpy(full_mask[:, :, np.newaxis]).to(device)
                gauss_mask = 1 - torch.from_numpy(gauss_mask[:, :, np.newaxis]).to(device)
                resized = resized.clip(0,255)
                
                full_img = imgs[i][idimg[j]].clone()
                full_img[bbx[1]:bbx[3], bbx[0]:bbx[2], :] = resized

                full_img = imgs[i][idimg[j]]*(1-gauss_mask) + full_img*gauss_mask
                # full_img = imgs[i][idimg[j]]*(1-full_mask) + full_img*full_mask
                # resized = (pre[trackid + j] * 255.).permute(1, 2, 0)
                imgs[i][idimg[j]] = full_img
                # plt.imshow(cv2.cvtColor((resized/255).cpu().numpy(),cv2.COLOR_BGR2RGB))
            except:
                print('emb_roi error:',bbx, resized.shape)
                import sys
                sys.exit()
        trackid += len(idimg)

    return imgs

def emb_roi2im(pickedimg, imgs, bbxs, pre, device,):
    trackid = 0
    width = imgs[0][0].shape[1]
    for i in range(len(pickedimg)):
        idimg = pickedimg[i]
        imgs[i] = imgs[i].float().to(device)
        for j in range(len(idimg)):
            bbx = bbxs[i][idimg[j]]
            if bbx[2] > width: bbx[2] = width
            if bbx[3] > width: bbx[3] = width
            resize2ori = transforms.Resize([bbx[3] - bbx[1], bbx[2] - bbx[0]],interpolation=InterpolationMode.BICUBIC)
            try:
                resized = resize2ori(pre[trackid + j] * 255.).permute(1, 2, 0)
                # resized = (pre[trackid + j] * 255.).permute(1, 2, 0)
                imgs[i][idimg[j]][bbx[1]:bbx[3], bbx[0]:bbx[2], :] = resized.clip(0,255)
                # plt.imshow(cv2.cvtColor((resized/255).cpu().numpy(),cv2.COLOR_BGR2RGB))
            except:
                print(bbx, resized.shape)
                import sys
                sys.exit()
        trackid += len(idimg)

    return imgs




def images2avhubert(pickedimg, imgs, bbxs, pre, video_size, device):
    imgs = emb_roi2im(pickedimg, imgs, bbxs, pre, device)
    processed_img = affine_trans(imgs, video_size).to(device)
    return processed_img


