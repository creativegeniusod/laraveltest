
import os
import cv2
import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import sys
# with PdfPages('multipage_pdf.pdf') as pdf:
# 	filepath = sys.argv[1]
# 	image = sys.argv[2]
# 	pdf_path = sys.argv[3]
# 	img = cv2.imread(filepath)
# 	blurred_img = cv2.blur(img, ksize=(4, 4))
# 	edges = cv2.Canny(image=blurred_img, threshold1=20, threshold2=60)
# 	plt.imshow(edges)
# 	status = plt.savefig(pdf_path, bbox_inches='tight')
# 	print status

 
with PdfPages(sys.argv[3]) as pdf:

   	# plt.xticks([])
   	# plt.yticks([])
   	# plt.plot(500,1000)
   	# plt.axis([200, 200, 2000, 1000])
	x = np.arange(10)
	y = np.arange(10)
	fig_width_cm = 21                         # A4 page
	fig_height_cm = 30
	inches_per_cm = 1 / 2.58               # Convert cm to inches	 
	fig_width = fig_width_cm * inches_per_cm # width in inches
	fig_height = fig_height_cm * inches_per_cm       # height in inches
	fig_size = [fig_width, fig_height] 
   	fig = plt.figure(figsize = fig_size)
   	plt.title('Test Image')
	pdf.savefig()  # saves the current figure into a pdf page
   	#plt.close()
   	y = np.arange(10)
	fig_width_cm = 21                         # A4 page
	fig_height_cm = 29.7
	inches_per_cm = 1 / 2.58               # Convert cm to inches
	fig_width = fig_width_cm * inches_per_cm # width in inches
	fig_height = fig_height_cm * inches_per_cm       # height in inches
   	plt.title('Test Image')
	fig_size = [fig_width, fig_height] 
   	fig = plt.figure(figsize = fig_size)
	filepath = sys.argv[1]
	image = sys.argv[2]
	pdf_path = sys.argv[3]
	img = cv2.imread(filepath)
	blurred_img = cv2.blur(img, ksize=(4, 4))
	edges = cv2.Canny(image=blurred_img, threshold1=20, threshold2=60)
	plt.imshow(edges)
	status = pdf.savefig()
	print (status)

