import numpy as np
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization,Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from create_dataset import get_labels



labels=get_labels("labels.txt")
labels=list(labels.keys())

x=np.load("x.npy")
y=np.load("y.npy")
y=to_categorical(y,num_classes=len(labels)).astype(int)


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.85,random_state=11)
x_val,x_test,y_val,y_test=train_test_split(x_test,y_test,test_size=0.75,random_state=11)
print(np.shape(x_test),np.shape(y_test),np.shape(x_train),np.shape(y_train),np.shape(x_val),np.shape(y_val))

# Define input shape
input_shape = (x_train.shape[1], x_train.shape[2])  
print("Input shape ",input_shape)


# Define LSTM model
model = Sequential()

model.add(Masking(mask_value=-99, input_shape=input_shape))
#model.add(Masking(mask_value=0))

model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))
    
model.add(LSTM(128, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.5))
    
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(len(labels), activation='softmax'))


# Load pre-trained weights
weights_path = 'weights.keras'
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print(f"Loaded weights from {weights_path}")
else:
    print("No pre-trained weights found, training from scratch")


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('weights.keras', monitor='val_loss', save_best_only=True)

model.fit(x_train, y_train, epochs=2000, verbose=1,validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint])
loss,accuracy=model.evaluate(x_test,y_test)
model.summary()

res=model.predict(x_test)
print(labels[np.argmax(res[11])])
print(res[0])
print(labels[np.argmax(y_test[11])])



