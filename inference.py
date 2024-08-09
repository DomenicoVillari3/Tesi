import numpy as np
import os
from keras.utils import to_categorical
import mediapipe as mp
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization,Masking,Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from create_dataset import get_labels
from mp_detection import draw_landmarks,place_landmarks,extract_landmarks
import cv2
import numpy as np
from model import create_model


print("LABELS")
labels=get_labels("labels.txt")
labels=list(labels.keys())
print("LABELS OTTENUTE")
input_shape=(131,171)
#input_shape=(148,171) #lettere
#input_shape =(134,171) 
model=create_model(input_shape,labels)
model.load_weights("weights.keras")
print("MODELLO CARICATO")

sequence=[] #64 frames
sentence=[] 
threshold=0.75

# Initialize the Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.7)

# Apri la webcam
cap = cv2.VideoCapture(0)

while True:
    # Leggi un frame dalla webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-processa il frame come hai fatto con i tuoi dati di addestramento
    # Ad esempio, potresti dover ridimensionare il frame, normalizzarlo, ecc.
    # Questo dipenderà dal tuo specifico pipeline di pre-processing
    frame,results=place_landmarks(frame,holistic)
    
    if results is not None:
        #Disegna landmarks
        draw_landmarks(frame=frame, results=results,mp_drawing=mp_drawing,mp_holistic=mp_holistic)
        
        # Estrazione dei landmarks su un array unidimensionale
        landmarks=extract_landmarks(results)

    else:
        landmarks=np.full(input_shape[1],-99)

    # Verifica se sono tutti NaN negli array di landmarks o se l'array è None, in tal caso skippo
    if  landmarks is None or np.all(np.isnan(landmarks)) :
        #print(":Non ci sono mani negli array di landmarks, salto questo frame.")
        landmarks=np.full(input_shape[1],-99)
    
    sequence.append(landmarks)
    
    sequence=sequence[-input_shape[0]:]

    if np.all(sequence == -99):
        sequence = []
        continue
    
    if len(sequence)==input_shape[0]:
        #passiamo 1 seq (1,64,162)
        res=model.predict(np.expand_dims(sequence,axis=0))[0]
        label=labels[np.argmax(res)]
        print(label)
        print(res[np.argmax(res)])
        if res[np.argmax(res)] >= threshold:
            if len(sentence)>0:
                if label != sentence[-1]:
                    sentence.append(label)
            else:
                sentence.append(label)
            
            if len(sentence)>5:
                sentence=sentence[-5:] 
            # Visualizza l'etichetta sul frame
            sequence=sequence[-input_shape[0]:]

        cv2.putText(frame, ' '.join(sentence), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostra il frame
    cv2.imshow('Webcam', frame)

    # Esci se l'utente preme 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la webcam e distrugge tutte le finestre
cap.release()
cv2.destroyAllWindows()
