import os 
import random
import numpy as np
import cv2

def capture_writer(input_path,output_path):
    # Leggi il video
    video = cv2.VideoCapture(input_path)

    # Ottieni le proprietà del video
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Crea un VideoWriter per scrivere il video modificato
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    return video,out


def add_random_noise(input_path, output_path):
    video,out=capture_writer(input_path,output_path)
    intensity=random.randint(15,60)
    while True:
        success, frame = video.read()
        
        if not success:
            break

        noisy_image = frame.copy()
        noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)

        noisy_frame = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)

        # Scrivi il frame nel nuovo video
        out.write(noisy_frame)
    
    # Rilascia le risorse
    video.release()
    out.release()


def grayscale(input_path,output_path):
    video,out=capture_writer(input_path,output_path)

    while True:
        success, frame = video.read()
        
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Scrivi il frame nel nuovo video
        out.write(gray)

    # Rilascia le risorse
    video.release()
    out.release()

    

def brightness_and_contrast(input_path,output_path):
    video,out=capture_writer(input_path,output_path)
    brightness=50
    contrast=1.5

    while True:
        success, frame = video.read()
        
        if not success:
            break

        # Aumenta o diminuisci luminosita 
        bright_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

        # Scrivi il frame nel nuovo video
        out.write(bright_frame)
    
    # Rilascia le risorse
    video.release()
    out.release()

def color_jitter(input_path,output_path):
    video,out=capture_writer(input_path,output_path)

    jitter_matrix = np.float32([[1, 0, 0, np.random.randint(-50, 50)], 
                                [0, 1, 0, np.random.randint(-50, 50)], 
                                [0, 0, 1, np.random.randint(-50, 50)]])
    
    while True:
        success, frame = video.read()
        
        if not success:
            break

        jittered_frame = cv2.transform(frame, jitter_matrix)
        
        # Assicuriamoci che i valori dei pixel rimangano nel range [0, 255]
        jittered_frame = np.clip(jittered_frame, 0, 255).astype(np.uint8)

        
        # Scrivi il frame nel nuovo video
        out.write(jittered_frame)

    # Rilascia le risorse
    video.release()
    out.release()

    


def flip(input_path, output_path,verso=0):
    video,out=capture_writer(input_path,output_path)

    while True:
        success, frame = video.read()
        
        if not success:
            break

        # Capovolgi il frame 
        frame = cv2.flip(frame, verso)

        # Scrivi il frame nel nuovo video
        out.write(frame)

    # Rilascia le risorse
    video.release()
    out.release()

def blur(input_path, output_path,vertical=True):
    video,out=capture_writer(input_path,output_path)

    # Specify the kernel size. 
    # The greater the size, the more the motion. 
    kernel_size = 5
    
    # Create the vertical kernel. 
    kernel_v = np.zeros((kernel_size, kernel_size)) 
    
    # Create a copy of the same for creating the horizontal kernel. 
    kernel_h = np.copy(kernel_v) 
    
    # Fill the middle row with ones. 
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
    
    # Normalize. 
    kernel_v /= kernel_size 
    kernel_h /= kernel_size 

    while True:
        success, frame = video.read()
            
        if not success:
            break
        
        if vertical:
            # Apply the vertical kernel. 
            blurred = cv2.filter2D(frame, -1, kernel_v) 
        else:
            # Apply the horizontal kernel. 
            blurred = cv2.filter2D(frame, -1, kernel_h)

        out.write(blurred)
    # Rilascia le risorse
    video.release()
    out.release()
    

def rotate(input_path, output_path):
    video,out=capture_writer(input_path,output_path)

    #Angolo di rotazione
    angle=random.randint(45,180)

    while True:
        success, frame = video.read()
        
        if not success:
            break
        
        #Recupero il centro del frame 
        (h, w) = frame.shape[:2]
        center = (w / 2, h / 2)

        #cv2.getRotationMatrix2D serve per creare la matrice di rotazione e prende 3 parametri:
        #1)centro di rotazione
        #2)angolo di rotazione
        #3) scala 
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        #eseguo la rotazione con il metodo warpAffine (applica una trasformazione )
        rotated = cv2.warpAffine(frame, M, (w, h))

        out.write(rotated)
    
    # Rilascia le risorse
    video.release()
    out.release()

def resize(input_path,output_path,scale=0.5):
    # Leggi il video
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    # Ottieni le proprietà del video
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Crea un VideoWriter per scrivere il video modificato
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(frame_width*scale),int( frame_height*scale)))


    while True:
        success, frame = video.read()
            
        if not success:
            break

        #resize image
        
       
    
        # resize image
        resized = cv2.resize(frame, (0,0), fx=scale, fy=scale) 

        out.write(resized)
    
    # Rilascia le risorse
    



    
def traslate(input_path,output_path):
    video,out=capture_writer(input_path,output_path)

    # Traslazione del frame
    x=random.randint(10,50)
    y=random.randint(-25,-1)

    if random.choice([True, False])==False:
        x=-x
    
    

    while True:
        success, frame = video.read()
        
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


        if not success:
            break

        #matrice di traslazione del frame
        M = np.float32([[1, 0, x], [0, 1, y]])

        # Applico la traslazione con 3 parametri
        #1) frame
        #2) matrice di traslazione del frame
        #3)dimensione del frame 
        trasleted = cv2.warpAffine(frame, M, (frame_width, frame_height))

        out.write(trasleted)

    # Rilascia le risorse
    video.release()
    out.release()

def change_fps(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return
 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    new_fps = original_fps * 2  # Nuovo FPS, raddoppiato

    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), new_fps, (width, height))
    
    prev_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if prev_frame is not None:
            #usa il frame precedente per duplicarlo
            output.write(prev_frame)
            output.write(frame)
        else:
            output.write(frame)
        
        prev_frame = frame
    
    cap.release()
    output.release()


    


def show_video(path):
    cap=cv2.VideoCapture(path)
    while True:
        success, frame = cap.read()
        
        if not success:
            break
        cv2.imshow("'Video'",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


inp="/home/domenico/tesi/video/acqua_0.mp4"
out="/home/domenico/tesi/video/acqua.mp4"

#blur(inp, out,False)
#flip(inp,out,0)
#traslate(inp,out)
#resize(inp,out,0.1)
#add_random_noise(inp,out)
#color_jitter(inp,out)
#change_fps(inp,out)
#show_video(inp)
#show_video(out)



