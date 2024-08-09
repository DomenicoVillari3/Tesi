import numpy as np
import os
import sys
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization,Masking,Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from create_dataset import get_labels
import matplotlib.pyplot as plt

def create_model(input_shape,labels):
    # Define LSTM model
    model = Sequential()

    model.add(Masking(mask_value=-99, input_shape=input_shape))
    #model.add(Masking(mask_value=0))

    #model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(73, return_sequences=False)))   
    #model.add(LSTM(73, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
        
    model.add(Dense(190, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(len(labels), activation='softmax'))
    return model

def evaluate_model(model,xtest,ytest,history,model_name=""):
    loss, acc = model.evaluate(xtest, ytest, verbose=0)
    print('Test Loss:', loss)
    print('Test Accuracy:', acc)

    # Plot training & validation accuracy values
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model accuracy '+model_name)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Confusion matrix
    prediction=model.predict(xtest)
    #recupero lista con gli indici corrispondenti ai risultati corretti nelle label
    ytrue=np.argmax(ytest,axis=1).tolist()
    ypredicted=np.argmax(prediction, axis=1).tolist()
    
    confusion_m=confusion_matrix(ytrue,ypredicted)
    display=ConfusionMatrixDisplay(confusion_m)
    display.plot()
    plt.show()






def main():
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

    model=create_model(input_shape,labels)



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

    history=model.fit(x_train, y_train, epochs=2000, verbose=1,validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint])
    model.summary()

    evaluate_model(model,x_test,y_test,history,"Coordinate Angolari")

    res=model.predict(x_test)
    print(labels[np.argmax(res[11])])
    #print(res[0])
    print(labels[np.argmax(y_test[11])])



if __name__ == '__main__':
    if len(sys.argv)==2:
        if sys.argv[1]=='-t' or sys.argv[1]=='-train':
            main()

        
    