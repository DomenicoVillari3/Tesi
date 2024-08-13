import cv2 
import mediapipe as mp
import numpy as np
import os 
import sys

from coordinate_sferiche import coordinate_sferiche
from get_angle import get_angle

# Initialize the Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5)
NUM_POINTS=1575
VIDEO_DIR = "/home/domenico/tesi/video"
POINTS_DIR_NAME="points"
FRAMES_FILENAME="frames.txt"



'''FUNZIONE PER UTILIZZARE MEDIAPIPE SULLA WEBCAM
    -Input=None
    -Output=None'''
def use_camera():
    #inizia cattura
    cap = cv2.VideoCapture(0)
     
    #finchè viene eseguita
    while cap.isOpened():

        # lettura del frame
        success, frame = cap.read()

        #condizione di uscita
        if not success:
            continue 

        # Predizione del modello
        frame, results = place_landmarks(frame, holistic)
        #print(results)
        
        if results is not None:
            #Disegna landmarks
            draw_landmarks(frame=frame, results=results,mp_drawing=mp_drawing,mp_holistic=mp_holistic)
            
            # Estrazione dei landmarks su un array unidimensionale
            if len(sys.argv)==3:
                if sys.argv[2]=="-a" or sys.argv[2]=='--angles':
                    landmarks=extract_landmarks_angles(results)
                else:
                    landmarks=extract_landmarks_points(results)
            else:
                landmarks=extract_landmarks_points(results)
            
            print(landmarks)
            #if landmarks is not None:
            #    print(len(landmarks))

            #normalizzazione dei landmarks tra [0,1]
            #normalized_landmarks=normalize_landmarks(landmarks)
            #print(normalized_landmarks)
            
        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


'''UTILIZZO DEL MODELLO DI MEDIAPIPE
    -Input=frame da valutare, modello di detection
    output= frame, risultati della detection
'''
def place_landmarks(frame,model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Converto il frame a RGB 
    frame.flags.writeable = False                           # Rendo il frame non scrivibile     
    results = model.process(frame)                      # Process
    frame.flags.writeable = True                            # Rendo il frame scrivibile     
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame,results


'''FUNZIONE PER DISEGNARE I LANDMARK
    Input=risultati del modello holistic, il frame analizzato,utility di drawing, holistic per ottenere le connessioni dei punti
    Output=None 
'''
def draw_landmarks(results,frame,mp_drawing,mp_holistic):
    
    mp_face_mesh = mp.solutions.face_mesh
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), 
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

    if results.pose_landmarks:
        #print(len(results.pose_landmarks.landmark))
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0),thickness=1),
                             mp_drawing.DrawingSpec(color=(0, 255, 0),thickness=1))
    
                                
    if results.left_hand_landmarks:
        #print(len(results.left_hand_landmarks.landmark))
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(255, 0, 0),thickness=1),
                                    mp_drawing.DrawingSpec(color=(255, 0, 0),thickness=1))
    if results.right_hand_landmarks:
        #print(len(results.left_hand_landmarks.landmark))
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,255),thickness=1),
                                    mp_drawing.DrawingSpec(color=(0,0,255),thickness=1))
                                  


'''Funzione per elaborare i landmarks in un array unidimensionale
    Input=risultati del modello holistic
    Output=array ad 1 dim con i landmark
'''

def extract_landmarks_angles(results):

    #LEFT HAND
    hand_left_array = []
    if results.left_hand_landmarks:
        
        ang1=get_angle((results.left_hand_landmarks.landmark[4].x,results.left_hand_landmarks.landmark[4].y),
                    (results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y),
                    (results.left_hand_landmarks.landmark[20].x,results.left_hand_landmarks.landmark[20].y))
        ang2=get_angle((results.left_hand_landmarks.landmark[8].x,results.left_hand_landmarks.landmark[8].y),
                    (results.left_hand_landmarks.landmark[5].x,results.left_hand_landmarks.landmark[5].y),
                    (results.left_hand_landmarks.landmark[12].x,results.left_hand_landmarks.landmark[12].y))
        ang3=get_angle((results.left_hand_landmarks.landmark[12].x,results.left_hand_landmarks.landmark[12].y),
                    (results.left_hand_landmarks.landmark[13].x,results.left_hand_landmarks.landmark[13].y),
                    (results.left_hand_landmarks.landmark[16].x,results.left_hand_landmarks.landmark[16].y))
        hand_left_array.append((ang1,ang2,ang3))
            
        hand_left_array= np.array(hand_left_array).flatten()
    else:
        #ANGOLI
        hand_left_array=np.zeros(3)  
    
    #RIGHT HAND
    hand_right_array = []
    if results.right_hand_landmarks:
        ang1=get_angle((results.right_hand_landmarks.landmark[4].x,results.right_hand_landmarks.landmark[4].y),
                    (results.right_hand_landmarks.landmark[0].x,results.right_hand_landmarks.landmark[0].y),
                    (results.right_hand_landmarks.landmark[20].x,results.right_hand_landmarks.landmark[20].y))
        ang2=get_angle((results.right_hand_landmarks.landmark[8].x,results.right_hand_landmarks.landmark[8].y),
                    (results.right_hand_landmarks.landmark[5].x,results.right_hand_landmarks.landmark[5].y),
                    (results.right_hand_landmarks.landmark[12].x,results.right_hand_landmarks.landmark[12].y))
        ang3=get_angle((results.right_hand_landmarks.landmark[12].x,results.right_hand_landmarks.landmark[12].y),
                    (results.right_hand_landmarks.landmark[13].x,results.right_hand_landmarks.landmark[13].y),
                    (results.right_hand_landmarks.landmark[16].x,results.right_hand_landmarks.landmark[16].y))
        hand_right_array.append((ang1,ang2,ang3))
            
        hand_right_array= np.array(hand_right_array).flatten()
    else:
        #ANGOLI
        hand_right_array=np.zeros(3)  
    
    ret=np.concatenate((hand_left_array, hand_right_array),axis=None) 
    if np.all(ret == 0) or len(ret) == 0 or len(ret)<6:
        return None
    else:
        return ret

    
    

def extract_landmarks_points(results):
    # Array per i landmarks delle pose e delle mani 3D, se non c'è creo un array di 0 
    # 33 punti per la posa 
    #21 per le mani 
    #Punti totali 1575 =(21*3)+(21*3)+(15*3)+(468*3)
    #1575

    #FACE MESH
    face_array = []
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            face_array.append((landmark.x, landmark.y, landmark.z))

            #rho,theta,phi=coordinate_sferiche(landmark.x, landmark.y, landmark.z)
            #face_array.append((rho,theta,phi))

        face_array=np.array(face_array).flatten()
        #print(len(face_array))
    #altrimenti riempio di 0
    else:
        face_array=np.zeros(468*3)
        

    pose_array=[]
    if results.pose_landmarks:
        for index,landmark in enumerate(results.pose_landmarks.landmark):
            #prendo solo i primi 15 punti [0,14]
            if index<=14:
                pose_array.append((landmark.x, landmark.y, landmark.z))

                #rho,theta,phi=coordinate_sferiche(landmark.x, landmark.y, landmark.z)
                #pose_array.append((rho,theta,phi))

        pose_array=np.array(pose_array).flatten()
        
    #altrimenti riempio di 0
    else:
        pose_array=np.zeros(15*3)


    #LEFT HAND
    hand_left_array = []
    #se viene rilevata la mano appendo le coordinate alla lista
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            hand_left_array.append((landmark.x, landmark.y, landmark.z))

            #rho,theta,phi=coordinate_sferiche(landmark.x, landmark.y, landmark.z)
            #hand_left_array.append((rho,theta,phi))

        hand_left_array=np.array(hand_left_array).flatten()
    #altrimenti riempio di 0
    else:
        hand_left_array=np.zeros(21*3)
    
  
    #RIGHT HAND
    hand_right_array = []
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            #POINTS
            hand_right_array.append((landmark.x, landmark.y, landmark.z))

            #COORDINATE ANGOLARI
            #rho,theta,phi=coordinate_sferiche(landmark.x, landmark.y, landmark.z)
            #hand_right_array.append((rho,theta,phi))

        hand_right_array= np.array(hand_right_array).flatten()
    else:
        #PUNTI E COORD ANGOLARI
        hand_right_array=np.zeros(21*3)
    
    #torno un array ad 1 dimensione (126,)
    ret=np.concatenate((hand_left_array, hand_right_array),axis=None) 

    #se ret contiene solo 0 non sono state rilevate mani, dunque il frame potrà essere scartato
    if np.all(ret == 0) or len(ret) == 0:
        return None
    else:
        return  np.concatenate((ret,pose_array),axis=None)
        #return  np.concatenate((ret,pose_array,face_array),axis=None)
        #return ret
    


'''FUNZIONE PER NORMALIZZARE I LANDMARKS CON NORMALIZZAZIONE MINMAX'''
def normalize_landmarks(landmarks):
    # Calcola i valori massimi e minimi per ciascuna dimensione
    min_vals = np.min(landmarks, axis=0)
    max_vals = np.max(landmarks, axis=0)
    #print(min_vals,max_vals)

    # Normalizza tra 0 e 1
    normalized_landmarks =(landmarks - min_vals) / (max_vals - min_vals) 

    return normalized_landmarks


'''FUNZIONE PER SALVARE I LANDMARK SU FILE .npy'''

def save_landmarks(landmarks, action):
    # Creo una directory ds
    dir = POINTS_DIR_NAME
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Salvo i landmarks
    filepath = dir + "/{}.npy".format(action)
    #print(filepath)
    np.save(filepath, landmarks)


'''
funzione per estrapolare i landmarks dai frame 
'''
def get_video_landmarks(video_path):
    #starting the video capture
    cap = cv2.VideoCapture(video_path)
    
   
    frame_index = 1
    all_landmarks = []

    #finchè è aperta la cattura dei frame
    while cap.isOpened():
        
        success, frame = cap.read()
        
        #se non va a buon fine la cattura del frame si esce
        if not success:
            break
        
        
        #utilizzo modello holistic
        frame, results = place_landmarks(frame, holistic)

        #se ho un risultato estraggo le coordinate dei landmarks, altrimenti imposto i landmark a None 
        if results is not None:
            
            '''draw_landmarks(frame=frame, results=results,mp_drawing=mp_drawing,mp_holistic=mp_holistic)
            # Show to screen
            cv2.imshow('OpenCV Feed', frame)
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break'''

            if len(sys.argv)==3:
                if sys.argv[2]=="-a" or sys.argv[2]=='--angles':
                    landmarks=extract_landmarks_angles(results)
                else:
                    landmarks=extract_landmarks_points(results)
            else:
                landmarks=extract_landmarks_points(results)
            
            #print(landmarks)
            #landmarks=normalize_landmarks(landmarks)
        else:
            landmarks=None
            

        # Verifica se sono tutti NaN negli array di landmarks o se l'array è None, in tal caso skippo
        if  landmarks is None or np.all(np.isnan(landmarks)) :
            #print(":Non ci sono mani negli array di landmarks, salto questo frame.")
            pass
        
        #Altrimenti verifico se ci sono nan ed eventualmente li sostituisco con 0, ed incremento il frame index
        else:
            nan_indices = np.isnan(landmarks)
            landmarks[nan_indices] = 0
            all_landmarks.append(landmarks)
            frame_index+=1
    
    cap.release()
    cv2.destroyAllWindows()

    return all_landmarks,frame_index


'''
Process Landmarks, prende in input la directory contenente i video e ne salva i landmark su file npy,
 crea il file con il count dei frame analizzati.
'''
def process_landmarks(dir):
    #lista per i frames
    frames=[]

    #lista per i video sotto la directory
    videos=os.listdir(dir)

    #scorro tutti i video 
    for i,video in enumerate(videos):
        print("video: ",i)
        #recupero l'azione descritta nel video
        action=video.split("_")
        action=action[0]
        

        #recupero l' il path del video 
        input_video_path=os.path.join(dir,video)

        # Get landmarks from video
        all_landmarks, frame_count = get_video_landmarks(input_video_path)

        #appendo il numero di frame nella lista (utile per avere contezza del numero di frame processati per ogni video)
        frames.append(frame_count)

        # Save landmarks to file
        save_landmarks(np.array(all_landmarks),video.replace('.mp4',''))
        print(f"Processed {video}")


    # Salvo il frame count su file 
    with open(FRAMES_FILENAME, "w") as file:
        file.write("\n".join(map(str, frames)))
        print("Frames count saved to frames.txt")



print(f"sys.argv: {sys.argv}")
if len(sys.argv)>= 1:
    if sys.argv[1] == '-p' or sys.argv[1] == '--process':
        process_landmarks(dir=VIDEO_DIR)

    elif sys.argv[1]=="-c" or sys.argv[1] == '--camera':
        use_camera()

    else:
        print(f"Unrecognized argument: {sys.argv[1]}")
else:
    print("No arguments provided.")
