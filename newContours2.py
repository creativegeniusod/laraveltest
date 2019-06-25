#!/usr/bin/env python
# coding: utf-8
# In[1]:

#Data path: in this direcory should reside the following files:
# - trained model file
# - xray image for the patient
# - box plot stats
# - pdf template file
import os
import sys
data_path = "/var/www/html/newfilelaravel/laraveltest/public/user-uploads/pythonscript/"

import pandas as pd, numpy as np

# In[2]:
#Trained model to do the reality check
reality_check_model_name = "reality_check_V2.h5"

#Trained model to use to predict keypoints
model_name = "NASNetMobile.h5"

#Patient's xray
patient_name = sys.argv[3]+"_"+sys.argv[2]
xray_file = sys.argv[1]

#Output
prediction_file = "prediction_for_%s.txt" % (patient_name)
output_file = "visualization_for_%s_TEMP.pdf" % (patient_name)
formatted_output = "%s.pdf" % (patient_name)

# In[3]:
patients_to_predict = pd.DataFrame(columns=(["image_name", "patient_name"]))

patients_to_predict["image_name"] = [data_path+xray_file]
patients_to_predict["patient_name"] = [patient_name]

# In[4]:
import keras
from keras import models, layers, backend
from keras.models import Sequential, Model

from keras_preprocessing.image import ImageDataGenerator

# In[5]:
## Reality Check
# In[6]:
model = models.load_model(data_path+reality_check_model_name)

# In[7]:
datagen = ImageDataGenerator(rescale=1./255)
reality_generator = datagen.flow_from_dataframe(
                color_mode = "rgb",
                dataframe = patients_to_predict,
                directory = None,
                x_col = "image_name",
                y_col = None,
                subset = None,
                batch_size = 1,
                seed = 0,
                shuffle = False,
                class_mode = None,
                target_size = (224, 224))

# In[8]:
STEP_SIZE_TEST = reality_generator.n//reality_generator.batch_size
reality_generator.reset()
preds = model.predict_generator(reality_generator, steps = STEP_SIZE_TEST, verbose=1)

# In[9]:
 
#if preds[0][0] > 0.5:
#    print("The current image %s is not recognisable. Please upload a cephalometry image" % (data_path+xray_file))
#    sys.exit()

keras.backend.clear_session()

# In[10]:
## Prediction Step

# In[11]:
model = models.load_model(data_path+model_name)
model.summary()

# In[ ]:
#Input and output definitions for the model

#original input_shape=(1935, 2400, 1)
#input dimensions divided by 4:
model_input_shape=(484, 600, 1)
#Number of keypoints * 2
num_outputs = 19*2

# In[ ]:
test_generator = datagen.flow_from_dataframe(
        color_mode = "grayscale",
        dataframe = patients_to_predict,
        directory = None,
        x_col = "image_name",
        y_col = None,
        subset = None,
        batch_size = 1,
        seed = 0,
        shuffle = False,
        class_mode = None,
        target_size = (model_input_shape[0], model_input_shape[1]))

# In[ ]:
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
test_generator.reset()
preds = model.predict_generator(test_generator, steps = STEP_SIZE_TEST, verbose=1)

#Image size
from PIL import Image
image_file = Image.open(data_path+xray_file, 'r')
image_shape = image_file.size

# In[ ]:
#Converting predictions to coordinates
#In the prediction vector, even indexes are for x and odd indexes for y
def pred_to_coord(pred_vector, image_size=(1935, 2400)):

    #The image size reference that was used for converting the targets to relative positions
    image_size_half = (image_size[0]/2, image_size[1]/2)

    return [int((p * image_size_half[0]) + image_size_half[0]) if i % 2 == 0 else int((p * image_size_half[1]) + image_size_half[1]) for i,p in enumerate(pred_vector)]

preds = [pred_to_coord(pred, image_size=image_shape) for pred in preds]

# In[ ]:

#Write the predictions in the output format
def write_prediction_to_file(prediction, file_name, file_path):
    out1 = np.array(prediction)
    out1 = out1.reshape(19, 2)
    out1 = pd.DataFrame(out1, columns=['x','y'])
    out1.to_csv(file_path+file_name, header=False, index=False)

# In[ ]:

for pred, p_name in zip(preds, patients_to_predict["patient_name"]):
    write_prediction_to_file(pred, prediction_file, data_path)

# In[ ]:

## Visualization Step

# In[ ]:

import matplotlib.pyplot as plt
import subprocess


# In[ ]:


plt.rcParams['figure.figsize'] = [11, 6]
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.weight'] = 'light'
plt.rcParams['font.size'] = 5
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5


# In[ ]:


df = pd.read_csv(data_path+'raw_values_senior.csv')
df2 = pd.read_csv(data_path+'raw_values_junior.csv')

df.columns = ['image', '1. ANB Value', '2. SNB Value', '3. SNA Value', '4. ODI Value',
              '5. APDI Value', '6. FHI Value', '7. FMA Value', '8. MW Value',
              '9. Palatal to Mandibular (VBT)', '10. Frankfort to Mandibular (VBT)',
              '11. Y-axis', '12. Subspinale to Nasion Perpendicular',
              '13. Pogonion to Nasion Perpendicular']
df2.columns = ['image', '1. ANB Value', '2. SNB Value', '3. SNA Value', '4. ODI Value',
               '5. APDI Value', '6. FHI Value', '7. FMA Value', '8. MW Value',
               '9. Palatal to Mandibular (VBT)', '10. Frankfort to Mandibular (VBT)',
               '11. Y-axis', '12. Subspinale to Nasion Perpendicular',
               '13. Pogonion to Nasion Perpendicular']


# In[ ]:


from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(data_path + output_file)


# In[ ]:


# Running angle.py on the predictions
s2_out = str(subprocess.check_output([sys.executable, data_path + "Angle.py", data_path + prediction_file]))
s2_out = s2_out[2:-1]
s2_out = s2_out.replace("\\n", "\n")
s2_out = s2_out.replace("\\r", "")
print(s2_out)


# In[ ]:


out_lines = s2_out.split('\n')[:-2]
row_names = []
cell_texts = []
for out_line in out_lines:
    cell_text = []
    value_name, out_line = out_line.split(': ')
    value, value_type = out_line.split(' Type:')
    value_type, type_text = value_type.split(' - ')
    row_names.append(value_name)
    cell_text.append(value[:7])
    cell_text.append(value_type)
    cell_text.append(type_text[:-1])
    cell_texts.append(cell_text)


# In[ ]:


# Visualizing the predicted keypoints
jpgfile = Image.open(data_path + xray_file, 'r')
np_im = np.asarray(jpgfile)
print(np_im.shape)

# Predicted coordinates
Seniorcoordinates = open(data_path + prediction_file, "r").readlines()
SX = []
SY = []
counter = 0
for i in Seniorcoordinates:
    if counter == 19:
        break
    i = i.replace("\n", "")
    X, Y = i.split(",")

    SX.append(int(X))
    SY.append(int(Y))
    counter = counter + 1

plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("CEPHALOMETRIC CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')

for i in range(len(SX)):
  plt.annotate(i + 1, (SX[i] + 40, SY[i] + 40), size=5, color='green', alpha=1.0, weight='light')

implot = plt.imshow(np_im)
plt.scatter(x=SX, y=SY, c='r', s=3)
plt.axis('on')

plt.subplot(2, 3, 3)
plt.title("SNAPSHOT CEPHALOMETRIC CLASSIFICATION SUMMARY", x=0.23)
the_table = plt.table(cellText=cell_texts,
                      rowLabels=row_names,
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      loc='center')
the_table.set_fontsize(7)
the_table.scale(1.2, 0.8)
plt.axis('off')

plt.subplot(2, 2, 4)
x = ['ANB', 'SNB', 'SNA', 'ODI', 'APDI', 'FHI', 'FMA', 'MW', "VBTM_A_type", "VBTM_B_type", "Yaxis_type", "MAXILLA_B_type", "MANDIBLE_B_type"]
Percentile_Value = [float(cell_texts[0][0]),
                    float(cell_texts[1][0]),
                    float(cell_texts[2][0]),
                    float(cell_texts[3][0]),
                    float(cell_texts[4][0]),
                    float(cell_texts[5][0]),
                    float(cell_texts[6][0]),
                    float(cell_texts[7][0]),
                    float(cell_texts[8][0]),
                    float(cell_texts[9][0]),
                    float(cell_texts[10][0]),
                    float(cell_texts[11][0]),
                    float(cell_texts[12][0]), ]

variance = [1, 2, 7, 4, 2, 3, 6, 8, 7, 4, 2, 3, 9]
x_pos = [i for i, _ in enumerate(x)]
plt.barh(x_pos, Percentile_Value, color='#ff6961', xerr=variance)
plt.ylabel("Classification")
# plt.xlabel("Percentile")
plt.title('Cephalometric Benchmark Summary', fontsize=7, fontweight='bold')
plt.yticks(x_pos, x)
pp.savefig()


# In[ ]:


#Reading the boxplot stats file
box_stats = pd.read_csv(data_path+"boxplot_stats.csv")

def get_box_stat(box_stats, value_to_get):
    stats = box_stats[box_stats['label'] == value_to_get].to_dict(orient='records')
    fliers = stats[0]['fliers']
    stats[0]['fliers'] = np.fromstring(fliers[1:-1], sep=' ')
    return stats


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

NX = [i for i in range(len(SX))]

def drawLine2P(x, y, delta=0.2, lin_color='#75DAFF'):
    dist = np.linalg.norm(np.array((x[0], y[0]))-np.array((x[1], y[1])))
    a1, b1 = np.polyfit(x, y, 1)
    x1 = min(x[0], x[1])
    x2 = max(x[0], x[1])
    y1 = min(y[0], y[1])
    y2 = max(y[0], y[1])
    x_delta = dist*delta if abs(a1)<1 else dist*delta/abs(a1)
    xlims = (max(0, x1 - x_delta), x2 + x_delta)
    xrange = np.arange(xlims[0], xlims[1], 0.1)
    plt.plot(xrange, (a1 * xrange) + b1, lin_color, alpha=1, linewidth=0.6)


# In[ ]:


#### 1 ####
#### ANB: the angle between Point 5 (vector 4), Point 2 (vector 1) and Point 6 (vector 5) ####
#### ANB = SNB- SNA
plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("CEPHALOMETRIC CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)
plt.scatter(x=[SX[1], SX[4], SX[5]], y=[SY[1], SY[4], SY[5]], c='r', alpha=1, s=3)
point_numbers = [NX[1], NX[4], NX[5]]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='green', alpha=1, weight='light')

# drawLine2P([SX[0], SX[1]], [SY[0], SY[1]])
plt.plot([SX[1], SX[4]], [SY[1], SY[4]], '#75DAFF', alpha=1, linewidth=0.6)
plt.plot([SX[1], SX[5]], [SY[1], SY[5]], '#75DAFF', alpha=1, linewidth=0.6)
# drawLine2P([SX[1], SX[5]], [SY[1], SY[5]])
#plt.annotate(" SNA=111 \n SNB=222 \n ANB=333", (SX[1] + 20, SY[1] + 10), color='white', alpha=1.0, weight='bold')

# plt.subplot(1, 2, 2)
# plt.annotate(out_lines[0], (0, 0.5), size=8)
# plt.axis('off')
plt.subplot(3, 2, 2)
plt.title("ANB ANGLE VALUE SUMMARY")
viz_number = 0
the_table = plt.table(cellText=[cell_texts[viz_number]],
                      rowLabels=[row_names[viz_number]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      loc='center')
# the_table.scale(1.05, 1.75)
the_table.set_fontsize(6)
the_table.scale(1, 1)
plt.axis('off')

# The Histogram (Probability Density Function)
axes = plt.subplot(3, 2, 4)


def histogram(df, df2, x_dim):
  x = df[x_dim]
  x2 = df2[x_dim]
  num_bins = 80
  axes.hist([x, x2], num_bins, alpha=0.60, density=True, histtype='barstacked', color=['#d55e00', '#607c8e'], label=['Senior Values', 'Junior Values'])
  plt.axvline(float(cell_texts[viz_number][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.0, ymax=0.97)
# adds a title and axes labels
  axes.set_title('Total Density Function\nSenior + Junior')
  axes.set_xlabel('')
  axes.set_ylabel('')
 # removing top and right borders
  axes.spines['top'].set_visible(False)
  axes.spines['right'].set_visible(False)
  axes.legend(loc='upper right')
  # adds major gridlines
  axes.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.4)


histogram(df, df2, '1. ANB Value')

# The Boxplot
axes = plt.subplot(3, 2, 6)
stats = get_box_stat(box_stats, row_names[viz_number])
axes.bxp(stats, vert=False)
plt.axvline(float(cell_texts[viz_number][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.03, ymax=0.97)

pp.savefig()


# In[ ]:


#### 2 ####
#### SNB: the angle between Point 1 (vector 0), Point 2 (vector 1) and Point 6 (vector 5) ####

plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("CEPHALOMETRIC CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)

plt.scatter(x=[SX[0], SX[1], SX[5]], y=[SY[0], SY[1], SY[5]], c='r', alpha=1, s=3)

point_numbers = [NX[0], NX[1], NX[5]]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='green', alpha=1.0, weight='light')

plt.plot([SX[0], SX[1]], [SY[0], SY[1]], '#75DAFF', alpha=1, linewidth=0.6)
#drawLine2P([SX[0], SX[1]], [SY[0], SY[1]])
plt.plot([SX[1], SX[5]], [SY[1], SY[5]], '#75DAFF', alpha=1, linewidth=0.6)
#drawLine2P([SX[1], SX[5]], [SY[1], SY[5]])

# plt.subplot(1, 2, 2)
# plt.annotate(out_lines[1], (0, 0.5), size=8)
# plt.axis('off')
plt.subplot(3, 2, 2)
plt.title("SNB ANGLE VALUE SUMMARY")
viz_number = 1
the_table = plt.table(cellText=[cell_texts[viz_number]],
                      rowLabels=[row_names[viz_number]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      loc='center')
the_table.set_fontsize(6)
the_table.scale(1, 1)
plt.axis('off')

# The Histogram (Probability Density Function)
axes = plt.subplot(3, 2, 4)
histogram(df, df2, '2. SNB Value')

# The Boxplot
axes = plt.subplot(3, 2, 6)
stats = get_box_stat(box_stats, row_names[viz_number])
axes.bxp(stats, vert=False)
plt.axvline(float(cell_texts[viz_number][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.03, ymax=0.97)

pp.savefig()


# In[ ]:


#### 3 ####
#### SNA: the angle between Point 1 (vector 0), Point 2 (vector 1) and Point 5 (vector 4) ####

plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("CEPHALOMETRIC CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)

plt.scatter(x=[SX[0], SX[1], SX[4]], y=[SY[0], SY[1], SY[4]], c='r', alpha=1, s=3)

point_numbers = [NX[0], NX[1], NX[4]]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='#75DAFF', alpha=1.0, weight='light')

plt.plot([SX[0], SX[1]], [SY[0], SY[1]], '#75DAFF', alpha=1, linewidth=0.6)
#drawLine2P([SX[0], SX[1]], [SY[0], SY[1]])
plt.plot([SX[1], SX[4]], [SY[1], SY[4]], '#75DAFF', alpha=1, linewidth=0.6)


plt.subplot(3, 2, 2)
plt.title("SNA ANGLE VALUE SUMMARY")
viz_number = 2
the_table = plt.table(cellText=[cell_texts[viz_number]],
                      rowLabels=[row_names[viz_number]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      loc='center')
the_table.set_fontsize(6)
the_table.scale(1, 1)
plt.axis('off')

# The Histogram (Probability Density Function)
axes = plt.subplot(3, 2, 4)
histogram(df, df2, '3. SNA Value')


# The Boxplot
axes = plt.subplot(3, 2, 6)
stats = get_box_stat(box_stats, row_names[viz_number])
axes.bxp(stats, vert=False)
plt.axvline(float(cell_texts[viz_number][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.03, ymax=0.97)

pp.savefig()


# In[ ]:


#### 4 ####
#### Overbite Depth Indicator ####
# The arithmetic sum of the angle between the AB plane (Point 5 (vector 4) to Point 6 (vector 5)) to the Mandibular Plane (MP, Point 8 (vector 7) to Point 10 (vector 9)) and the angle of the Palatal Plane (PP, Point 17 (vector 16) to Point 18 (vector 17)) to Frankfort Horizontal plane (FH, Point 4 (vector 3) to Point 3 (vector 2)).
# (Vectors include: 2, 3, 16, 17, 4, 5, 7, 9)

plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("CEPHALOMETRIC CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)

plt.scatter(x=SX[2:6] + [SX[7], SX[9]] + SX[16:18], y=SY[2:6] + [SY[7], SY[9]] + SY[16:18], c='r', alpha=1, s=3)

point_numbers = NX[2:6] + [NX[7], NX[9]] + NX[16:18]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='green', alpha=1.0, weight='light')

#plt.plot([SX[9], SX[7]], [SY[9], SY[7]], 'g-', alpha=1, linewidth=0.6)
drawLine2P([SX[9], SX[7]], [SY[9], SY[7]])
drawLine2P([SX[5], SX[4]], [SY[5], SY[4]])
# plt.plot([SX[4], SX[5]], [SY[4], SY[5]], 'g-', alpha=1, linewidth=0.6)
plt.plot([SX[3], SX[2]], [SY[3], SY[2]], '#75DAFF', alpha=1, linewidth=0.6)
plt.plot([SX[16], SX[17]], [SY[16], SY[17]], 'orange', alpha=1, linewidth=0.6)

plt.subplot(3, 2, 2)
plt.title("OVERBITE DEPTH INDICATOR VALUE SUMMARY")
viz_number = 3
the_table = plt.table(cellText=[cell_texts[viz_number]],
                      rowLabels=[row_names[viz_number]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      loc='center')
the_table.set_fontsize(6)
the_table.scale(1, 1)
plt.axis('off')

# The Histogram (Probability Density Function)
axes = plt.subplot(3, 2, 4)
histogram(df, df2, '4. ODI Value')

# The Boxplot
axes = plt.subplot(3, 2, 6)
stats = get_box_stat(box_stats, row_names[viz_number])
axes.bxp(stats, vert=False)
plt.axvline(float(cell_texts[viz_number][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.03, ymax=0.97)

pp.savefig()


# In[ ]:


#### 5 ####
#### Anteroposterior dysplasia indicator ####
#### (Vectors include: 2, 3, 1, 6, 4, 5, 16, 17) ####

plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("CEPHALOMETRIC CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)

plt.scatter(x=SX[1:7] + SX[16:18], y=SY[1:7] + SY[16:18], c='r', alpha=1, s=3)

point_numbers = NX[1:7] + NX[16:18]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='green', alpha=1.0, weight='light')

plt.plot([SX[2], SX[3]], [SY[2], SY[3]], '#75DAFF', alpha=1, linewidth=0.6)
plt.plot([SX[1], SX[6]], [SY[1], SY[6]], '#75DAFF', alpha=1, linewidth=0.6)
#plt.plot([SX[4], SX[5]], [SY[4], SY[5]], 'g-', alpha=1, linewidth=0.6)
drawLine2P([SX[5], SX[4]], [SY[5], SY[4]])
plt.plot([SX[3], SX[2]], [SY[3], SY[2]], '#75DAFF', alpha=1, linewidth=0.6)
drawLine2P([SX[3], SX[2]], [SY[3], SY[2]])
plt.plot([SX[16], SX[17]], [SY[16], SY[17]], 'orange', alpha=1, linewidth=0.6)

plt.subplot(3, 2, 2)
plt.title("ANTEROPOSTERIOR DYSPLASIA INDICATOR VALUE SUMMARY")
viz_number = 4
the_table = plt.table(cellText=[cell_texts[viz_number]],
                      rowLabels=[row_names[viz_number]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      loc='center')
the_table.set_fontsize(6)
the_table.scale(1, 1)
plt.axis('off')

# The Histogram (Probability Density Function)
axes = plt.subplot(3, 2, 4)
histogram(df, df2, '5. APDI Value')

# The Boxplot
axes = plt.subplot(3, 2, 6)
stats = get_box_stat(box_stats, row_names[viz_number])
axes.bxp(stats, vert=False)
plt.axvline(float(cell_texts[viz_number][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.03, ymax=0.97)

pp.savefig()


# In[ ]:


#### 6 ####
# FHI, obtained by the ratio of the Posterior Face Height (PFH = the distance from Point 1 (vector 0) to Point 10 (vector 9)) to the Anterior Face Height (AFH = the distance from Point 2 (vector 1) to Point 8 (vector 7)). FHI = PFH / AFH.
#### (Vectors include: 0, 9, 1, 7) ####
plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("CEPHALOMETRIC CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)

plt.scatter(x=SX[0:2] + [SX[7], SX[9]], y=SY[0:2] + [SY[7], SY[9]], c='r', alpha=1, s=3)

point_numbers = NX[0:2] + [NX[7], NX[9]]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='green', alpha=1.0, weight='light')

plt.plot([SX[0], SX[9]], [SY[0], SY[9]], '#75DAFF', alpha=1, linewidth=0.6)
plt.plot([SX[1], SX[7]], [SY[1], SY[7]], '#75DAFF', alpha=1, linewidth=0.6)

plt.subplot(3, 2, 2)
plt.title("FACE HEIGHT INDICATOR SUMMARY")
viz_number = 5
the_table = plt.table(cellText=[cell_texts[viz_number]],
                      rowLabels=[row_names[viz_number]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      loc='center')
the_table.set_fontsize(6)
the_table.scale(1, 1)
plt.axis('off')

# The Histogram (Probability Density Function)
axes = plt.subplot(3, 2, 4)
histogram(df, df2, '6. FHI Value')


# The Boxplot
axes = plt.subplot(3, 2, 6)
stats = get_box_stat(box_stats, row_names[viz_number])
axes.bxp(stats, vert=False)
plt.axvline(float(cell_texts[viz_number][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.03, ymax=0.97)

pp.savefig()


# In[ ]:


#### 7 ####
# FMA: the angle between the line from sella (Point 1)(vector 0) to nasion (Point 2)(vector 1) and the line from gonion (Point 10)(vector 9) to gnathion (Point 9)(vector 8)
#### (Vectors include: 0, 1, 8, 9) ####
plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("CEPHALOMETRIC CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)

plt.scatter(x=SX[0:2] + SX[8:10], y=SY[0:2] + SY[8:10], c='r', alpha=1, s=3)

point_numbers = NX[0:2] + NX[8:10]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='green', alpha=1.0, weight='light')

# plt.plot([SX[0], SX[1]], [SY[0], SY[1]], 'g-', alpha=1, linewidth=0.6)
# plt.plot([SX[8], SX[9]], [SY[8], SY[9]], 'g-', alpha=1, linewidth=0.6)
drawLine2P([SX[0], SX[1]], [SY[0], SY[1]])
drawLine2P([SX[9], SX[8]], [SY[9], SY[8]])

plt.subplot(3, 2, 2)
plt.title("FMA VALUE SUMMARY")
viz_number = 6
the_table = plt.table(cellText=[cell_texts[viz_number]],
                      rowLabels=[row_names[viz_number]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      loc='center')
the_table.set_fontsize(6)
the_table.scale(1, 1)
plt.axis('off')

# The Histogram (Probability Density Function)
axes = plt.subplot(3, 2, 4)
histogram(df, df2, '7. FMA Value')

# The Boxplot
axes = plt.subplot(3, 2, 6)
stats = get_box_stat(box_stats, row_names[viz_number])
axes.bxp(stats, vert=False)
plt.axvline(float(cell_texts[viz_number][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.03, ymax=0.97)

pp.savefig()


# In[ ]:


#### 8 ####
# Modify-Wits (MW): the distance between upper (Point 12) and lower (Point 11). MW = square root((x_{L12} - x_{L11})2 + (y_{L12} - y_{L11})2), if x_{L12} > x_{L11}, a positive MW; otherwise, a negative MW.
#### (Vectors include: 10, 11) ####
plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("CEPHALOMETRIC CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)

plt.scatter(x=SX[10:12], y=SY[10:12], c='r', alpha=1, s=3)

point_numbers = NX[10:12]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='green', alpha=1.0, weight='light')

drawLine2P([SX[10], SX[11]], [SY[10], SY[11]])

plt.subplot(3, 2, 2)
plt.title("MODIFIED WITS APPRAISAL VALUE SUMMARY")
viz_number = 7
the_table = plt.table(cellText=[cell_texts[viz_number]],
                      rowLabels=[row_names[viz_number]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      loc='center')
the_table.set_fontsize(6)
the_table.scale(1, 1)
plt.axis('off')

# The Histogram (Probability Density Function)
axes = plt.subplot(3, 2, 4)
histogram(df, df2, '8. MW Value')

# The Boxplot
axes = plt.subplot(3, 2, 6)
stats = get_box_stat(box_stats, row_names[viz_number])
axes.bxp(stats, vert=False)
plt.axvline(float(cell_texts[viz_number][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.03, ymax=0.97)


pp.savefig()


# In[ ]:


#### 9 and 10 ####
#### Vertical Bite Tendency Measure ####
# Palatal Plane (ANS [L17] - PNS [L18]) to Mandibular Plan Angle (Go [L10] - Me [L8])
# Angle of Frankfort Horizontal (Por [L4] - O [L3]) to Mandibular Plane (Go [L10] - Me [L8])
plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("VERTICAL BITE CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)

plt.scatter(x=[SX[2], SX[3], SX[7], SX[9], SX[16], SX[17]], y=[SY[2], SY[3], SY[7], SY[9], SY[16], SY[17]], c='r', alpha=1, s=3)
point_numbers = [NX[2], NX[3], NX[7], NX[9], NX[16], NX[17]]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='green', alpha=1, weight='light')

# Palatal Plane (ANS [L17] - PNS [L18]) to Mandibular Plan Angle (Go [L10] - Me [L8])
drawLine2P([SX[9], SX[7]], [SY[9], SY[7]])
drawLine2P([SX[16], SX[17]], [SY[16], SY[17]])
# plt.plot([SX[2], SX[3]], [SY[2], SY[3]], '#75DAFF', alpha=1, linewidth=0.6)

# Angle of Frankfort Horizontal (Por [L4] - O [L3]) to Mandibular Plane (Go [L10] - Me [L8])
plt.plot([SX[2], SX[3]], [SY[2], SY[3]], '#75DAFF', alpha=1, linewidth=0.6)

plt.subplot(3, 3, 3)
plt.title("VERTICAL BITE CLASSIFICATION SUMMARY", x=0.23)
viz_number1 = 8
viz_number2 = 9
the_table = plt.table(cellText=[cell_texts[viz_number1], cell_texts[viz_number2]],
                      rowLabels=[row_names[viz_number1], row_names[viz_number2]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      loc='center')
# the_table.scale(1.05, 1.75)
the_table.set_fontsize(6)
the_table.scale(1.2, 0.8)
plt.axis('off')


# The Histogram (Probability Density Function)
axes = plt.subplot(3, 2, 4)
histogram(df, df2, '9. Palatal to Mandibular (VBT)')

# The Boxplot
axes = plt.subplot(3, 2, 6)
# stats1 = get_box_stat(box_stats, row_names[viz_number1])
# stats2 = get_box_stat(box_stats, row_names[viz_number2])
# stats = stats1 + stats2
# axes.bxp(stats, vert=False)
# plt.axvline(float(cell_texts[viz_number1][0]))
# plt.axvline(float(cell_texts[viz_number2][0]))

histogram(df, df2, '10. Frankfort to Mandibular (VBT)')

pp.savefig()


# In[ ]:


#### 11 ####
#### Skeletal Growth Pattern ####
# Angle of Frankfort Horizontal (Por [L4] - O [L3]) to (Sella [L1] - Gnathion [L8])

plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("SKELETAL GROWTH CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)

plt.scatter(x=[SX[0], SX[2], SX[3], SX[7]], y=[SY[0], SY[2], SY[3], SY[7]], c='r', alpha=1, s=3)
point_numbers = [NX[0], NX[2], NX[3], NX[7]]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='green', alpha=1, weight='light')

# Angle of Frankfort Horizontal (Por [L4] - O [L3]) to (Sella [L1] - Gnathion [L8])
plt.plot([SX[0], SX[7]], [SY[0], SY[7]], '#75DAFF', alpha=1, linewidth=0.6)
plt.plot([SX[2], SX[3]], [SY[2], SY[3]], '#75DAFF', alpha=1, linewidth=0.6)

plt.subplot(3, 2, 2)
plt.title("SKELETAL GROWTH CLASSIFICATION SUMMARY")
viz_number = 10
the_table = plt.table(cellText=[cell_texts[viz_number]],
                      rowLabels=[row_names[viz_number]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      loc='center')
# the_table.scale(1.05, 1.75)
the_table.set_fontsize(6)
the_table.scale(1, 1)
plt.axis('off')

axes = plt.subplot(3, 2, 4)
histogram(df, df2, '11. Y-axis')

# The Boxplot
axes = plt.subplot(3, 2, 6)
stats = get_box_stat(box_stats, row_names[viz_number])
axes.bxp(stats, vert=False)
plt.axvline(float(cell_texts[viz_number][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.03, ymax=0.97)


pp.savefig()


# In[ ]:


#Paints the Nasion Perpendicular and its distance to a point (in a dashed line)

def distance_to_nasion_perpendicular(point_to_compare_index, lin_color='yellow'):
    #Frankfurt Line equation
    x = [SX[3], SX[2]]
    y = [-SY[3], -SY[2]]
    a1,b1 = np.polyfit(x, y, 1)
    #Nasion Perpendicular equation
    a = -(1.0 / a1)
    b = -SY[1] - (a*SX[1])
    #X at the Y of the point
    x2 = (-SY[point_to_compare_index] - b) / a

    drawLine2P([SX[1], x2], [SY[1], SY[point_to_compare_index]], lin_color=lin_color)
    plt.plot([SX[point_to_compare_index], x2], [SY[point_to_compare_index], SY[point_to_compare_index]],
             lin_color, linewidth=0.5, linestyle='--')


# In[ ]:


#### 12 ####
#### THE MAXILLA ####
#### SNA: the angle between Point 1 (vetor 0), Point 2 (vector 1) and Point 5 (vector 4) ####
#### Distance between the point A [L5] to the Nasion Perpendicular (NP) ####

plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("THE MAXILLA VALUE SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)

plt.scatter(x=[SX[0], SX[1], SX[2], SX[3], SX[4]], y=[SY[0], SY[1], SY[2], SY[3], SY[4]], c='r', alpha=1, s=3)
point_numbers = [NX[0], NX[1], NX[2], NX[3], NX[4]]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='green', alpha=1, weight='light')

#### SNA: the angle between Point 1 (vetor 0), Point 2 (vector 1) and Point 5 (vector 4) ####
plt.plot([SX[0], SX[1]], [SY[0], SY[1]], '#75DAFF', alpha=1, linewidth=0.6)
plt.plot([SX[1], SX[4]], [SY[1], SY[4]], '#75DAFF', alpha=1, linewidth=0.6)

# Distance between the point A [L5] to the Nasion Perpendicular (NP)
drawLine2P([SX[3], SX[2]], [SY[3], SY[2]])
# NAISON PERPENDICULAR LINE GOES HERE -------------
# DOTTED LINE from NAISON PERPENDICULAR TO POINT 5 GOES HERE -------------
distance_to_nasion_perpendicular(4)


plt.subplot(3, 3, 3)
plt.title("THE MAXILLA CLASSIFICATION SUMMARY", x=0.23)
viz_number1 = 11
viz_number2 = 2

the_table = plt.table(cellText=[cell_texts[viz_number1], cell_texts[viz_number2]],
                      rowLabels=[row_names[viz_number1], row_names[viz_number2]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      rowColours='grey',
                      loc='center')
# the_table.scale(1.05, 1.75)
the_table.set_fontsize(6)
the_table.scale(1.2, 0.8)
plt.axis('off')

# The Histogram (Probability Density Function)
axes = plt.subplot(3, 2, 4)
histogram(df, df2, '12. Subspinale to Nasion Perpendicular')

# The Boxplot
axes = plt.subplot(3, 2, 6)
# stats1 = get_box_stat(box_stats, row_names[viz_number1])
stats = get_box_stat(box_stats, row_names[viz_number1])
# stats = stats1 + stats2
axes.bxp(stats, vert=False)
axes.set_yticklabels([])
axes.set_xlabel('')
axes.set_ylabel('12. Subspinale to\n Nasion\n Perpendicular')
# plt.axvline(float(cell_texts[viz_number1][0]))
plt.axvline(float(cell_texts[viz_number1][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.03, ymax=0.97)

pp.savefig()


# In[ ]:


#### 13 ###
#### THE MANDIBLE ####
#### SNB: the angle between Point 1 (vector 0), Point 2 (vector 1) and Point 6 (vector 5) ####
# Distance between the Pogonion [L16] to the Nasion Perpendicular (NP).

plt.figure()
plt.subplot(1, 2, 1)
text = ("Patient " + xray_file + " " + prediction_file)
plt.text(400, -120, text, size=8, fontweight='bold', ha='left')
plt.title("THE MANDIBLE CLASSIFICATION SUMMARY")
plt.annotate("Patient " + xray_file + " " + prediction_file, (SX[1] - 500, SY[1] + 1300), size=6, color='white', alpha=0.4, weight='light')
implot = plt.imshow(np_im)

plt.scatter(x=[SX[0], SX[1], SX[2], SX[3], SX[5], SX[15]], y=[SY[0], SY[1], SY[2], SY[3], SY[5], SY[15]], c='r', alpha=1, s=3)
point_numbers = [NX[0], NX[1], NX[2], NX[3], NX[5], NX[15]]
for i in point_numbers:
  plt.annotate(i + 1, (SX[i] + 20, SY[i] + 0), color='green', alpha=1, weight='light')

#### SNB: the angle between Point 1 (vector 0), Point 2 (vector 1) and Point 6 (vector 5) ####
plt.plot([SX[0], SX[1]], [SY[0], SY[1]], '#75DAFF', alpha=1, linewidth=0.6)
plt.plot([SX[1], SX[5]], [SY[1], SY[5]], '#75DAFF', alpha=1, linewidth=0.6)

# Distance between the Pogonion [L16] to the Nasion Perpendicular (NP).
drawLine2P([SX[3], SX[2]], [SY[3], SY[2]])

# NAISON PERPENDICULAR LINE GOES HERE -------------
# DOTTED LINE from NAISON PERPENDICULAR TO POINT 15 GOES HERE -------------
distance_to_nasion_perpendicular(15)

plt.subplot(3, 3, 3)
plt.title("THE MANDIBLE CLASSIFICATION SUMMARY", x=0.23)
viz_number1 = 12
viz_number2 = 1

the_table = plt.table(cellText=[cell_texts[viz_number1], cell_texts[viz_number2]],
                      rowLabels=[row_names[viz_number1], row_names[viz_number2]],
                      colLabels=['Value', 'Type', 'Notes'],
                      colWidths=[0.15, 0.15, 0.7],
                      edges='horizontal',
                      rowColours='grey',
                      loc='center')
the_table.set_fontsize(6)
the_table.scale(1.2, 0.8)
plt.axis('off')

# The Histogram (Probability Density Function)
axes = plt.subplot(3, 2, 4)
histogram(df, df2, '13. Pogonion to Nasion Perpendicular')

# The Boxplot
axes = plt.subplot(3, 2, 6)
# stats1 = get_box_stat(box_stats, row_names[viz_number1])
stats = get_box_stat(box_stats, row_names[viz_number1])
# stats = stats1 + stats2
axes.bxp(stats, vert=False)
axes.set_yticklabels([])
axes.set_xlabel('')
axes.set_ylabel('13. Pogonion to\n Nasion\n Perpendicular')
# plt.axvline(float(cell_texts[viz_number1][0]))
plt.axvline(float(cell_texts[viz_number1][0]), color='black', linestyle='-.', linewidth=0.8, alpha=0.8, ymin=0.03, ymax=0.97)

pp.savefig()


# In[ ]:


pp.close()


# In[ ]:


from PyPDF2 import PdfFileWriter, PdfFileReader

# read the existing PDF
new_pdf = PdfFileReader(data_path + output_file)
existing_pdf = PdfFileReader(data_path + "original.pdf")
output = PdfFileWriter()

# add the "watermark" (which is the new pdf) on the existing page
page = existing_pdf.getPage(0)
output.addPage(page)
for i in range(13):
  page = existing_pdf.getPage(i + 1)
  page.mergeTranslatedPage(new_pdf.getPage(i), 0, 25, expand=False)
  output.addPage(page)

# finally, write "output" to a real file
outputStream = open(data_path + formatted_output, "wb")
status=output.write(outputStream)
print(status)
#outputStream.close()

