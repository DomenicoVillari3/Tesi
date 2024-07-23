import os 
import random
import numpy as np
import cv2
from argument import grayscale,flip,brightness_and_contrast,rotate,traslate,add_random_noise,show_video,blur,resize,color_jitter,change_fps
import gc

VIDEO_DIR = "video"

def argument_videos(dir,labels_file="labels.txt"):
    labels=[]

    videos=os.listdir(dir)
    for video in videos:
        input_video_path=os.path.join(dir,video)
        print(input_video_path)

        #recupero il nome del file 
        video=video.split("_")
        number=video[1].split(".")[0]
        video=video[0]
        
        #print(number)
        #print(video)
        labels.append(video)

        number=int(number)
    
        # Argumentation 1: Flip orizzontalmente
        number+=1
        out_video_path = os.path.join(dir, video + "_{}.mp4".format(number))
        flip(input_video_path, out_video_path,1)


        # Argumentation 2: Add random noise
        number+=1
        for i in range(number,number+5):
            out_video_path = os.path.join(dir, video + "_" + str(i) + ".mp4")
            #su metà dei video uso la mano opposta a quella del video
            if i%2==0:
                temp="temp.mp4"
                flip(input_video_path, temp,1)
                add_random_noise(temp, out_video_path)
            else:
                add_random_noise(input_video_path, out_video_path)
        number+=5

        # Argumentation 3: Increase brightness and contrast
        number+=1
        out_video_path = os.path.join(dir, video + "_{}.mp4".format(number))
        brightness_and_contrast(input_video_path, out_video_path)

        # Argumentation 4: Blur verticale
        number+=1
        out_video_path = os.path.join(dir, video + "_{}.mp4".format(number))
        blur(input_video_path, out_video_path,vertical=True)

        # Argumentation 5: Blur verticale
        number+=1
        out_video_path = os.path.join(dir, video + "_{}.mp4".format(number))
        blur(input_video_path, out_video_path,vertical=False)

        #Argumentation 6: raddoppio il numero di fps
        out_video_path = os.path.join(dir, video + "_{}.mp4".format(number))
        change_fps(input_video_path,out_video_path)

        # Argumentation 7: resizing
        number+=1
        for i in range(number,number+5):
            size=random.uniform(1.2, 1.8)
            out_video_path = os.path.join(dir, video + "_" + str(i) + ".mp4")
            #su metà dei video uso la mano opposta a quella del video
            if i%2==0:
                temp="temp.mp4"
                flip(input_video_path, temp,1)
                resize(input_video_path, out_video_path,size)
            else:
                resize(input_video_path, out_video_path,size)
        number+=5

        # Argumentation 8: resizing
        number+=1
        for i in range(number,number+5):
            size=random.uniform(0.2, 0.7)
            out_video_path = os.path.join(dir, video + "_" + str(i) + ".mp4")
            #su metà dei video uso la mano opposta a quella del video
            if i%2==0:
                temp="temp.mp4"
                flip(input_video_path, temp,1)
                resize(input_video_path, out_video_path,size)
            else:
                resize(input_video_path, out_video_path,size)
        number+=5

       
        # Argumentation 11: Color jitter (eseguito 5 volte)
        number+=1
        for i in range(number,number+5):
            out_video_path = os.path.join(dir, video + "_" + str(i) + ".mp4")
            #su metà dei video uso la mano opposta a quella del video
            if i%2==0:
                temp="temp.mp4"
                flip(input_video_path, temp,1)
                color_jitter(temp, out_video_path)
            else:
                color_jitter(input_video_path, out_video_path)
        number+=5
                
        
        # Argumentation 12: Rotation (eseguita 5 volte)
        for i in range(number,number+5):
            out_video_path = os.path.join(dir, video + "_" + str(i) + ".mp4")
            if i%2==1:
                temp="temp.mp4"
                flip(input_video_path, temp,1)
                rotate(temp, out_video_path)
            else:
                rotate(input_video_path, out_video_path)
        number+=5


        #Argumentation 8: Traslation (eseguita 5 volte)
        for i in range(number,number+5):
            out_video_path = os.path.join(dir, video + "_" + str(i) + ".mp4")
            if i%2==0:
                temp="temp.mp4"
                flip(input_video_path, temp,1)
                traslate(temp, out_video_path)
            else:
                traslate(input_video_path, out_video_path)
        gc.collect()
        number+=5
       

    
    if os.path.exists("temp.mp4"):
        os.remove("temp.mp4")
        

    with open(labels_file, 'w') as f:
        for label in set(labels):
            f.write(label + "\n")

        

labels=argument_videos(VIDEO_DIR)


