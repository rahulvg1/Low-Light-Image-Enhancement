
import numpy as np
from scipy import ndimage
import cv2


im = cv2.imread('img.png',0)


data = np.array(im, dtype=float)


hkernel = np.array([[-1, -1, -1, -1, -1],
                   [-1, -1, -1, -1, -1],
                   [-1, -1, 24, -1, -1],
                   [-1, -1, -1, -1, -1],
                   [-1, -1, -1, -1, -1]])    
    

lkernel = np.ones((5, 5), np.float32) / 25
    
highpass = ndimage.convolve(data, hkernel)
lowpass = ndimage.convolve(data, lkernel)


cv2.imwrite("high.png", highpass)
cv2.imwrite("low.png", lowpass)

