import os 
import random
import numpy as np
import cv2
from argument import grayscale,flip,brightness_and_contrast,rotate,traslate,add_random_noise,show_video

VIDEO_DIR = "/home/domenico/tesi/video"

def argument_videos(dir,labels_file="labels.txt"):
    labels=[]

    videos=os.listdir(dir)
    for video in videos:
        input_video_path=os.path.join(dir,video)

        #recupero il nome del file 
        video=video.split("_0")
        print(video[0])
        video=video[0]

        labels.append(video)
        

        # Augmentation 1: Flip
        out_video_path = os.path.join(dir, video + "_1.mp4")
        flip(input_video_path, out_video_path)

        # Augmentation 2: Rotation
        out_video_path = os.path.join(dir, video + "_2.mp4")
        rotate(input_video_path, out_video_path)

        #Augmentation 3: Traslation
        out_video_path = os.path.join(dir, video + "_3.mp4")
        traslate(input_video_path, out_video_path)

        # Augmentation 4: Add random noise
        out_video_path = os.path.join(dir, video + "_4.mp4")
        add_random_noise(input_video_path, out_video_path)

        # Augmentation 5: Increase brightness and contrast
        out_video_path = os.path.join(dir, video + "_5.mp4")
        brightness_and_contrast(input_video_path, out_video_path)

        # Augmentation 6: Gray scale
        out_video_path = os.path.join(dir, video + "_6.mp4")
        grayscale(input_video_path, out_video_path)

        for i in range(7,17):
            out_video_path = os.path.join(dir, video + "_" + str(i) + ".mp4")
            rotate(input_video_path, out_video_path)
        
        for i in range(17,21):
            out_video_path = os.path.join(dir, video + "_" + str(i) + ".mp4")
            traslate(input_video_path, out_video_path)

    with open(labels_file, 'w') as f:
        for label in labels:
            f.write(label + "\n")

        

labels=argument_videos(VIDEO_DIR)


