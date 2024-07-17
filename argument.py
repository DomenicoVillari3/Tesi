import os 
import random
import numpy as np
import cv2



def add_random_noise(input_path, output_path):
    # Leggi il video
    video = cv2.VideoCapture(input_path)

    # Ottieni le proprietà del video
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Crea un VideoWriter per scrivere il video modificato
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    intensity=random.randint(25,90)

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
    # Leggi il video
    video = cv2.VideoCapture(input_path)

    # Ottieni le proprietà del video
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Crea un VideoWriter per scrivere il video modificato
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height),isColor=False)

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
    # Leggi il video
    video = cv2.VideoCapture(input_path)

    # Ottieni le proprietà del video
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Crea un VideoWriter per scrivere il video modificato
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

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

    


def flip(input_path, output_path):
    # Leggi il video
    video = cv2.VideoCapture(input_path)

    # Ottieni le proprietà del video
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Crea un VideoWriter per scrivere il video modificato
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        success, frame = video.read()
        
        if not success:
            break

        # Capovolgi il frame erticalmente
        frame = cv2.flip(frame, 0)

        # Scrivi il frame nel nuovo video
        out.write(frame)

    # Rilascia le risorse
    video.release()
    out.release()


def rotate(input_path, output_path):
    # Leggi il video
    video = cv2.VideoCapture(input_path)

    # Ottieni le proprietà del video
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Crea un VideoWriter per scrivere il video modificato
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

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




    
def traslate(input_path,output_path):
    # Leggi il video
    video = cv2.VideoCapture(input_path)

    # Ottieni le proprietà del video
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Crea un VideoWriter per scrivere il video modificato
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Traslazione del frame
    x=random.randint(10,15)
    y=random.randint(-15,-1)

 

    while True:
        success, frame = video.read()
        
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


