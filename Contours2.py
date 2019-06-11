import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileWriter, PdfFileReader
from matplotlib.backends.backend_pdf import PdfPages
import sys

pdf_path = sys.argv[6]
image_file = sys.argv[1]
temp_pdf_file = sys.argv[4]+"_temp_holder.pdf"
formatted_output = sys.argv[3]
img = cv2.imread(image_file)
enhance_img = cv2.blur(img, ksize=(3, 3))
edges = cv2.Canny(image=enhance_img, threshold1=30, threshold2=60)
pp = PdfPages(pdf_path + temp_pdf_file)
plt.imshow(edges)
pp.savefig()
pp.close()

new_pdf = PdfFileReader(pdf_path + temp_pdf_file)
existing_pdf = PdfFileReader(sys.argv[5] + "original.pdf")
output = PdfFileWriter()
page = existing_pdf.getPage(0)
output.addPage(page)
for i in range(1):
    page = existing_pdf.getPage(i + 1)
    page.mergeTranslatedPage(new_pdf.getPage(i), 0, 25, expand=False)
    output.addPage(page)


outputStream = open(formatted_output, "wb")
status=output.write(outputStream)
print(status)

