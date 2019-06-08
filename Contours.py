#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/var/www/html/media/img-desktop-kv-background.jpg')
blurred_img = cv2.blur(img, ksize=(4, 4))
edges = cv2.Canny(image=blurred_img, threshold1=20, threshold2=60)

plt.imshow(edges)
plt.show()
