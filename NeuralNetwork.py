import tensorflow as tf
import matplotlib.pyplot as plt
import time 
import math
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization, Masking
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from create_dataset import get_labels
labels=get_labels("labels.txt")
labels=list(labels.keys())


class NeuralNetwork:
    def __init__(self, n_lstm, d_lstm, n_densi, d_densi):
        self.labels = get_labels("labels.txt")
        self.labels = np.array(self.labels)
        print(n_lstm, d_lstm, n_densi, d_densi)
        self.model = self.build_model(n_lstm, d_lstm, n_densi, d_densi)
    
    def load_data(self):
        x = np.load("x.npy")
        y = np.load("y.npy")
        y=to_categorical(y,num_classes=len(labels)).astype(int)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80)
        #print(f"shapes: \n xtrain {x_train.shape}, ytrain {y_train.shape} \n x_test {x_test.shape} ytest {y_test.shape}")

        
        return (x_train, y_train), (x_test, y_test)

    def build_model(self, n_strati_conv, dim_strati_conv, n_strati_densi, dim_strati_densi):
        input_shape = (82, 165)

        model = Sequential()
        model.add(Masking(mask_value=-99, input_shape=input_shape))

        for _ in range(n_strati_conv - 1):
            model.add(LSTM(dim_strati_conv, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
    

        
        model.add(LSTM(dim_strati_conv, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        for _ in range(n_strati_densi):
            model.add(Dense(dim_strati_densi, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))


        model.add(Dense(len(labels), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        
        return model
    
    def getModel(self):
        return self.model

    def train(self, x_train, y_train, epochs=50):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        start_time = time.time()
        history = self.model.fit(x=x_train, y=y_train,epochs=epochs, verbose=1, callbacks=[early_stopping], validation_split=0.1)
        end_time = time.time()
        training_time = end_time - start_time
        return history.history["categorical_accuracy"], training_time
    
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)


'''
if __name__ == "__main__":
    rete = NeuralNetwork(8, 40, 5, 256)
    (x_train, y_train), (x_test, y_test) = rete.load_data()
    model = rete.getModel()
    accuracy, training_time = rete.train(x_train, y_train, epochs=5)

    test_loss, test_accuracy = rete.evaluate(x_test, y_test)
    print("\n\n\n\n Accuratezza per epoca:", accuracy)
    print("Tempo di addestramento:", training_time)
    print("\n accuracy finale:", test_accuracy)
    print("FITNESS:", (test_accuracy / math.log10(training_time)) )
'''