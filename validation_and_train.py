# -*- coding: utf-8 -*-
"""validation and train.ipynb


"""

from google.colab import drive
drive.mount('/content/drive')

import os

path='/content/drive/MyDrive/Yolo_v4'
os.chdir(path)

!git clone https://github.com/AlexeyAB/darknet

os.chdir('/content/drive/MyDrive/Yolo_v4/darknet')
!make

!./darket

#Testing the YOLOv4 model if it works on general datasets before training for custom dataset.
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg

# Commented out IPython magic to ensure Python compatibility.
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
image=cv2.imread('predictions.jpg')
fig=plt.gcf()
fig.set_size_inches(12,14)
plt.imshow(image)

from google.colab import files
files.download('predictions.jpg')

!./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show data/.mp4 -i 0 -out_filename obj_det_video.avi

from google.colab import files
files.download('obj_det_video.avi')

#Train
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

os.chdir('/content/drive/MyDrive/Yolo_v4/darknet')
!sudo chmod +x darknet
!./darknet

!./darknet detector train data/Scoreboard/image_data.data cfg/yolov4_train.cfg yolov4.conv.137 -dont_show

!./darknet detector train data/Scoreboard/image_data.data cfg/yolov4_train.cfg /content/drive/MyDrive/Yolo_v4 /darknet/backup/yolov4_train_final.weights -dont_show

# Commented out IPython magic to ensure Python compatibility.
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

!./darknet detector test data/Scoreboard/image_data.data cfg/yolov4_train.cfg /content/drive/MyDrive/Yolo_v4/darknet/backup/yolov4_train_last.weights /content/drive/MyDrive/Yolo_v4/darknet/data/test_data/testing.jpg -thresh 0.3 -out predictions.txt

image=cv2.imread('predictions.jpg')
fig=plt.gcf()
fig.set_size_inches(14,12)
plt.imshow(image)

# Commented out IPython magic to ensure Python compatibility.
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

!./darknet detector test data/Scoreboard/image_data.data cfg/yolov4_train.cfg /content/drive/MyDrive/Yolo_v4/darknet/backup/yolov4_train_last.weights /content/drive/MyDrive/Yolo_v4/darknet/data/test_data/scoreboard_test2.jpg -thresh 0.3 -out predictions.txt

image=cv2.imread('predictions.jpg')
fig=plt.gcf()
fig.set_size_inches(14,12)
plt.imshow(image)

############################





