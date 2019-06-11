#!/usr/bin/env python
import os
#os.makedirs('/var/www/html/media/this_is_folders')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

filepath = sys.argv[1]
image = sys.argv[2]
pdf_path = sys.argv[3]

#print image

img = cv2.imread(filepath)
blurred_img = cv2.blur(img, ksize=(4, 4))
edges = cv2.Canny(image=blurred_img, threshold1=20, threshold2=60)

# storing image path
#filename = os.path.join('/webroot/public_html/websites/odz/laraveltest/public/user-uploads/pdf',image)

# save image
#status = cv2.imwrite('public/user-uploads/pdf/5cfada8b87aa0.jpg',edges)

#return status
#print status

#plt.imshow(edges)
#plt.show()

plt.imshow(edges)
status = plt.savefig(pdf_path, bbox_inches='tight')

print status
