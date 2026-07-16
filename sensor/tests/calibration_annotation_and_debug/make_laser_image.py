import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import re
import random
def number(p):
    match = re.search(r'\d+', p)
    return int(match.group()) if match else -1
folder = r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\"
photos = sorted(os.listdir(folder), key = number)
empty_path = folder + photos[2418-149]
if __name__ == "__main__":
   img = np.ones((640, 480), dtype=np.uint8) * 127
   laser = np.zeros((640, 640), dtype=np.uint8) * 255
   x = 160
   w = 80
   l = 325
   y = 210
   laser[x:x+w, y:y+l] = 255
   rotate = cv2.getRotationMatrix2D((320, 240), -(270 + 26.2), 1)
   laser = cv2.warpAffine(laser, rotate, (640, 480))
   a, b = random.sample(range(0, len(photos)), 2) 
   path = folder + photos[2270]
   im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
   print(laser.shape, im.shape)
   im[~(laser > 0)] = 0
   plt.subplot(1, 3, 1)
   plt.imshow(im, cmap='gray')
   plt.subplot(1, 3, 2)
   im2 = cv2.imread(folder + photos[b], cv2.IMREAD_GRAYSCALE)
   
   im2[~(laser > 0)] = 0
   plt.imshow(im2, cmap='gray')
   plt.title(f"{a, b}")
   plt.subplot(1, 3, 3)
   plt.imshow((im + im2)/2, cmap='gray')
   plt.show()