import cv2
import time
import numpy as np
import tqdm
im = np.random.randint(-1 ,300,(1920,1080,3)).astype(np.uint8)
s = time.time()
image=cv2.imread('max.jpg').astype(np.float32)
for i in tqdm.tqdm(range(1000)):
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    normalized_image = (normalized_image * 255).astype(np.uint8)
    # lab_img = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    # lab_img[:, :, 0] = cv2.equalizeHist(lab_img[:, :, 0])
    # processed_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
cv2.imwrite('pres.jpg',normalized_image)
e = time.time()
print(e-s)