import cv2
import numpy as np



def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # generate Gaussian pyramid for A,B and mask
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
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        gm = gm[:,:,np.newaxis]
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_


def test2():
    # 创建纯黑图像作为图片A
    image_a = np.zeros((512, 512, 3), dtype=np.uint8)

    # 创建纯白图像作为图片B
    image_b = np.ones((512, 512, 3), dtype=np.uint8) * 255

    image_a = cv2.imread('image_a.png')
    image_b = cv2.imread('image_b.png')
    image_a = cv2.resize(image_a,(512,512))
    image_b = cv2.resize(image_b,(512,512))


    # 创建掩码图像
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[128:384, 128:384] = 255

    image_a = image_a / 255.0
    image_b = image_b / 255.0
    mask = mask / 255.0

    # 进行图像融合
    result = Laplacian_Pyramid_Blending_with_mask(image_a, image_b, mask)

    # 将结果缩放回0-255之间的整数范围
    result = (result * 255).astype(np.uint8)
    cv2.imwrite('result.jpg',result)
test1()