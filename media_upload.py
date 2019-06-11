import os
import cv2
import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import sys
 
with PdfPages(sys.argv[3]) as pdf:
   fig_width_cm = 21
   fig_height_cm = 30
   inches_per_cm = 1 / 2.58
   fig_width = fig_width_cm * inches_per_cm
   fig_height = fig_height_cm * inches_per_cm
   fig_size = [fig_width, fig_height]
   fig = plt.figure(figsize = fig_size)
   plt.axis('off')
   plt.title('Test Image')
   pdf.savefig()
   
   fig_width_cm = 21
   fig_height_cm = 29.7
   inches_per_cm = 1 / 2.58
   fig_width = fig_width_cm * inches_per_cm
   fig_height = fig_height_cm * inches_per_cm
   fig_size = [fig_width, fig_height]
   fig = plt.figure(figsize = fig_size)
   filepath = sys.argv[1]
   image = sys.argv[2]
   pdf_path = sys.argv[3]
   # xmin xmax ymin ymax
   img = cv2.imread(filepath)
   blurred_img = cv2.blur(img, ksize=(4, 4))
   edges = cv2.Canny(image=blurred_img, threshold1=20, threshold2=60) 
   plt.imshow(edges)
   status = pdf.savefig()
   print (status)
