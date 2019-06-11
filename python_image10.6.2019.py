
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

   	plt.xticks([])
   	plt.yticks([])
   	plt.axis('off')
   	plt.title('Test Image')
   	pdf.savefig()  # saves the current figure into a pdf page
   	plt.close()
	filepath = sys.argv[1]
	image = sys.argv[2]
	pdf_path = sys.argv[3]
	img = cv2.imread(filepath)
	blurred_img = cv2.blur(img, ksize=(4, 4))
	edges = cv2.Canny(image=blurred_img, threshold1=20, threshold2=60)
	plt.imshow(edges)
	status = pdf.savefig()
	print (status)
   	# x = np.arange(0, 5, 0.1)
   	# plt.rc('text', usetex=False)
   	# fig = plt.figure(figsize=(4, 5))
   	# plt.plot(x, x ** 2, 'ko')
   	# plt.title('OUTPUT')
   	# stauu=pdf.savefig(fig)


