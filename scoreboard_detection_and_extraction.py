import zipfile
import os

# Defined the paths to the zip files
data_zip_path = "data.zip"  
model_zip_path = "model.zip"  
framess_zip_path = "frames_porspa4.zip" 
extract_dir = os.getcwd()

folder_name="frames_porspa4"
# Function to unzip files
def unzip_file(zip_path, extract_to):
    """
    Unzips the specified zip file to the target directory.
    
    :param zip_path: Path to the zip file.
    :param extract_to: Directory where the contents should be extracted.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

# # # Unzip the data.zip file
# unzip_file(data_zip_path, extract_dir)

# # Unzip the model.zip file
# unzip_file(model_zip_path, extract_dir)
# unzip_file(framess_zip_path, extract_dir)


import cv2
import os

def extract_frames_from_time_range(video_path, output_folder, frame_rate=1):
    """
    Extract frames from the start to the end of a video at a specified frame rate.

    Parameters:
    - video_path (str): Path to the input video file.
    - output_folder (str): Folder to save the extracted frames.
    - frame_rate (int): Number of frames to save per second of video.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Get the video's frame rate and duration
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    # Set the start and end times
    start_time = 0
    end_time = video_duration

    # Calculate start and end frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Set the video to start at the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened() and frame_count <= end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        # Only save frames according to the specified frame rate
        if frame_count % int(fps // frame_rate) == 0:
            frame_filename = f"{output_folder}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print("Frames extracted successfully.")


video_path = 'second_porspa.mp4'
output_folder = 'framess'
frame_rate = 1  # Extract 1 frame per second

# extract_frames_from_time_range(video_path, output_folder, frame_rate)





import cv2
import torch
import random
import collections
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as transforms

class strLabelConverter(object):

    def __init__(self, alphabet):

        alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'

        self.dict = {}

        for i, char in enumerate(alphabet):

            self.dict[char] = i + 1      # Position 0 is for space character

    def encode(self, text):

        # Encoding the word into integer format.
        text = [self.dict[char.lower()]for char in text]
        length = [len(text)]
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):

        length = length[0]
        assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
        if raw:
            return ''.join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return ''.join(char_list)

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self):

        super(CRNN, self).__init__()

        cnn = nn.Sequential()
        cnn.add_module('conv0',      nn.Conv2d(1, 64, 3, 1, 1))  # Input, Output, Kernal, Stride, Padding
        cnn.add_module('relu0',      nn.ReLU(True))
        cnn.add_module('pooling0',   nn.MaxPool2d(2, 2))                     # 64x16x64
        cnn.add_module('conv1',      nn.Conv2d(64, 128, 3, 1, 1))
        cnn.add_module('relu1',      nn.ReLU(True))
        cnn.add_module('pooling1',   nn.MaxPool2d(2, 2))                     # 128x8x32
        cnn.add_module('conv2',      nn.Conv2d(128, 256, 3, 1, 1))
        cnn.add_module('batchnorm2', nn.BatchNorm2d(256))
        cnn.add_module('relu2',      nn.ReLU(True))
        cnn.add_module('conv3',      nn.Conv2d(256, 256, 3, 1, 1))
        cnn.add_module('relu3',      nn.ReLU(True))
        cnn.add_module('pooling2',   nn.MaxPool2d((2, 2), (2, 1), (0, 1)))   # 256x4x16
        cnn.add_module('conv4',      nn.Conv2d(256, 512, 3, 1, 1))
        cnn.add_module('batchnorm4', nn.BatchNorm2d(512))
        cnn.add_module('relu4',      nn.ReLU(True))
        cnn.add_module('conv5',      nn.Conv2d(512, 512, 3, 1, 1))
        cnn.add_module('relu5',      nn.ReLU(True))
        cnn.add_module('pooling3',   nn.MaxPool2d((2, 2), (2, 1), (0, 1)))   # 512x2x16
        cnn.add_module('conv6',      nn.Conv2d(512, 512, 2, 1, 0))           # 512x1x16
        cnn.add_module('batchnorm6', nn.BatchNorm2d(512))
        cnn.add_module('relu6',      nn.ReLU(True))

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),       # Input, Hidden, Output
            BidirectionalLSTM(256, 256, 37))        # Final output: 37 classes

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)

        return output

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

def sort_by_frame_number(filenames):
    # Extract numbers from filenames and sort based on them
    return sorted(filenames, key=lambda x: int(re.search(r'(\d+)', x).group()))

def load_image(name):
    image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1280, 720))
    return image

import torch
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import csv
import re
# Initialize model, transformer, and other necessary components
def initialize_model():
    model_path = './model/crnn.pth'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    model = CRNN()

    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    converter = strLabelConverter(alphabet)
    transformer = resizeNormalize((100, 32))
    return model, converter, transformer

# Prediction function
def predict(model, converter, transformer, image_path):
    image = Image.open(image_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred

# Load image and prepare the cropped sections for OCR
def load_image(name):
    image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1280, 720))
    return image

def crop(box, name, img):
    crop_img = img[box[0]:box[1], box[2]:box[3]]
    cv2.imwrite('data/' + name, crop_img)





if(folder_name == 'frames'):
    team1_box = [43, 70, 110, 170]
    score_box = [43, 70, 180, 250]
    team2_box = [43, 70, 260, 320]
    time_box = [43, 70, 365, 430]
elif (folder_name == 'framess'):
    team1_box = [30, 80, 90, 160]
    score_box = [30, 80, 170, 240]
    team2_box= [40, 60, 260, 310]
    time_box = [30, 60, 360, 430]
else:
    team1_box = [43, 80, 110, 170]
    score_box = [30, 80, 180, 250]
    team2_box= [40, 80, 260, 310]
    time_box = [43, 70, 370, 430]






# Write timestamp to CSV file
def write_timestamp_to_csv(time, csv_file):
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        
        writer = csv.writer(file)
        if not file_exists or os.stat(csv_file).st_size == 0:
            writer.writerow(["Timestamp"])
        writer.writerow([time])
# Sort function to ensure filenames like "frame_1" are sorted correctly

# Main function to process multiple images
def process_images(image_folder):
    model, converter, transformer = initialize_model()
    previous_score=None
    # previous_time=None

    image_files = sort_by_frame_number(os.listdir(image_folder))
    for idx, image_name in enumerate(image_files):
        # print(image_name)
        image_path = os.path.join(image_folder, image_name)
        image = load_image(image_path)
        # print(previous_score)
        crop(team1_box, "team1.png", image)
        crop(score_box, "score.png", image)
        crop(time_box, "time.png", image)
        crop(team2_box, "team2.png", image)

        time = predict(model, converter, transformer, "data/time.png")
        team1 = predict(model, converter, transformer, "data/team1.png")
        team2 = predict(model, converter, transformer, "data/team2.png")
        score = predict(model, converter, transformer, "data/score.png")
        # print(time)
        # print(len(time))
        # Formatting time and score as required
        if(len(time))==3:
          time=time[:1]+time[1:]
        if len(time) == 4 :
          time = time[:2] +":"+ time[2:]
        if len(time)==5:
          time = time[:2] + ":" + time[3:]
        if len(time) == 3:
          time = time[:1] + ":" + time[1:]

        # if len(time) == 5:
        #     time = time[:2] + time[3:]
        # time = time[:2] + ":" + time[2:]

        if len(score) == 3:
          score = score[:1] + score[2:]
        
        if score[:1].isdigit() and score[1:].isdigit() and int(score[:1])<=4 :
          score = score[:1] + "-" + score[1:]
        else:
          score = "!-!"
        # print("Image:", image_name)
        # print(time)
        # print(score)
        # print("Time:       ", time)
        # print("Team 1:     ", team1)
        # print("Team 2:     ", team2)
        # print("Score:      ", score)
        # print("#############")
        # print(len(score))
        # Print the first frame details or whenever there's a change in the score/time
        # print(time)
        # print(previous_score)
        if(previous_score==None and score!="!-!" and len(score)==3 ):
            print("Image:", image_name)
            print("Time:       ", time)
            print("Team 1:     ", team1)
            print("Team 2:     ", team2)
            print("Score:      ", score)
            minutes = int(time.split(":")[0])
            # print(minutes)
            if minutes >=45:
                write_timestamp_to_csv(str(time),'por_spatimestamp1.csv')
            else:
                write_timestamp_to_csv(str(time),'por_spatimestamp2.csv')
            # print("Minutes: ", minutes)
            # write_timestamp_to_csv(str(time))
            previous_score = score
        elif (score != previous_score and (score!="!-!" and len(score)==3 )):

            print("Image:", image_name)
            print("Time:       ", time)
            print("Team 1:     ", team1)
            print("Team 2:     ", team2)
            print("Score:      ", score)
            
            minutes = int(time.split(":")[0])
            # print("Minutes: ", minutes)
            if minutes >=45:
                write_timestamp_to_csv(str(time),'por_spatimestamp1.csv')
            else:
                write_timestamp_to_csv(str(time),'por_spatimestamp2.csv')
            # print("Minutes: ", minutes)
            # Write timestamp to CSV
            # write_timestamp_to_csv(str(time))
            previous_score = score
        # if score!="!-!":
          # previous_score = score

        # Update previous score and time




process_images(folder_name)

















import time
time.sleep(1200)