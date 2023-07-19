import numpy as np
import cv2
def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # generate Gaussian pyramid for A,B and mask
    height, width = A.shape[:2]
    B = cv2.resize(B, (width, height))
    m = cv2.resize(m, (width, height))

    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    
    gpA = [np.float32(GA)]
    gpB = [np.float32(GB)]
    gpM = [np.float32(GM)]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.resize(cv2.pyrUp(gpA[i]),(gpA[i-1].shape[1],gpA[i-1].shape[0])))
        LB = np.subtract(gpB[i-1], cv2.resize(cv2.pyrUp(gpB[i]),(gpB[i-1].shape[1],gpB[i-1].shape[0])))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        # name= 'mask'+str(len(LS))+'.png'
        # cv2.imwrite(name,gm*255.0)
        gm = gm[:,:,np.newaxis]
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv2.add(ls_, LS[i])
    return ls_

def inpatient(new_imgs, ref_imgs,bbxs):
    results = []

    for new_img,ref_img,bbx in zip(new_imgs, ref_imgs,bbxs):
        x1,y1,x2,y2 = bbx.cpu()
        height, width = ref_img.shape[:2]

        mask = np.zeros_like(ref_img[:,:,0].cpu(), dtype='float32')
        mask[y1:y2,x1:x2] = 255
        new_img = new_img / 255.0
        ref_img = ref_img / 255.0
        mask = mask / 255.0
        new_img = new_img.cpu().detach().numpy()
        ref_img = ref_img.cpu().detach().numpy()
        if max(height,width) > 512:
            h,w = 1024,1024
        else:
            h,w=512,512
        new_img, ref_img , mask  = [cv2.resize(x, (h, w)) for x in (new_img, ref_img, np.float32(mask))]
        result = Laplacian_Pyramid_Blending_with_mask(new_img, ref_img, mask,10)
        result = np.uint8(cv2.resize(np.clip(result*255, 0 ,255), (width, height)))
        results.append(result)

    return results 

def test1():
    new_img = cv2.imread("image_a.png")
    new_img = cv2.resize(new_img,(480,854))
    ref_img = cv2.imread("image_b.png")
    ref_img = cv2.resize(ref_img,(480,854))
    height, width = ref_img.shape[:2]
    m = np.zeros_like(new_img[:,:,0], dtype='float32')
    m[:,int(new_img.shape[1]/2):] = 1
    if max(new_img.shape[0],new_img.shape[1]) > 512:
        h,w = 1024,1024
    else:
        h,w=512,512
    new_img, ref_img , m  = [cv2.resize(x, (h, w)) for x in (new_img, ref_img, np.float32(m))]
    result = Laplacian_Pyramid_Blending_with_mask(new_img, ref_img, m, 6)
    result = np.uint8(cv2.resize(np.clip(result, 0 ,255), (width, height)))
    cv2.imwrite("result2.png",result)


# test1()