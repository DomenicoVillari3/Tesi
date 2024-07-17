import os 
import random
import numpy as np
import cv2
from argument import grayscale,flip,brightness_and_contrast,rotate,traslate,add_random_noise,show_video,blur,resize,color_jitter


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
        
        # Argumentation 1: Flip verticalmente
        out_video_path = os.path.join(dir, video + "_1.mp4")
        flip(input_video_path, out_video_path)

        # Argumentation 2: Flip orizzontalmente
        out_video_path = os.path.join(dir, video + "_2.mp4")
        flip(input_video_path, out_video_path,1)

        # Argumentation 3: Flip orizzontalmente e verticalmente
        out_video_path = os.path.join(dir, video + "_3.mp4")
        flip(input_video_path, out_video_path,-1)

        # Argumentation 4: Add random noise
        out_video_path = os.path.join(dir, video + "_4.mp4")
        add_random_noise(input_video_path, out_video_path)

        # Argumentation 5: Gray scale
        out_video_path = os.path.join(dir, video + "_5.mp4")
        grayscale(input_video_path, out_video_path)

        # Argumentation 6: Increase brightness and contrast
        out_video_path = os.path.join(dir, video + "_6.mp4")
        brightness_and_contrast(input_video_path, out_video_path)

        # Argumentation 7: Blur verticale
        out_video_path = os.path.join(dir, video + "_7.mp4")
        blur(input_video_path, out_video_path,vertical=True)

        # Argumentation 8: Blur verticale
        out_video_path = os.path.join(dir, video + "_8.mp4")
        blur(input_video_path, out_video_path,vertical=False)

        # Argumentation 9: Resize con riduzione del 50%
        out_video_path=os.path.join(dir, video + "_9.mp4")
        resize(input_video_path, out_video_path,0.5)

        # Argumentation 10: Resize con aumento del 50%
        out_video_path=os.path.join(dir, video + "_10.mp4")
        resize(input_video_path, out_video_path,1.5)

        # Argumentation : Color jitter (eseguito 5 volte)
        for i in range(11,16):
            out_video_path = os.path.join(dir, video + "_" + str(i) + ".mp4")
            color_jitter(input_video_path, out_video_path)

        # Argumentation 16->19: Rotation (eseguita 5 volte)
        for i in range(16,20):
            out_video_path = os.path.join(dir, video + "_" + str(i) + ".mp4")
            rotate(input_video_path, out_video_path)
        
        #Argumentation 8: Traslation (eseguita 5 volte)
        for i in range(20,25):
            out_video_path = os.path.join(dir, video + "_" + str(i) + ".mp4")
            traslate(input_video_path, out_video_path)

    with open(labels_file, 'w') as f:
        for label in labels:
            f.write(label + "\n")

        

labels=argument_videos(VIDEO_DIR)


