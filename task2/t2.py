
import cv2
import numpy as np
import image_slicer
from image_slicer import join
from PIL import Image


img = 'img.jpg'
num_tiles = 16

tiles = image_slicer.slice(img, num_tiles)


for tile in tiles:
    img = cv2.imread(tile.filename, 0)
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf *hist.max()/ cdf.max()  
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_o = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdff = np.ma.filled(cdf_o,0)
    img2 = cdff[img]
    cv2.imwrite(tile.filename,img2)
    tile.image = Image.open(tile.filename)

image = join(tiles)
image.save('clahe.png')


img = cv2.imread("img.jpg",0)
clahe = cv2.imread("clahe.png",0)

cv2.imshow("Original",img)
cv2.imshow("Clahe",clahe)
cv2.waitKey(0)



