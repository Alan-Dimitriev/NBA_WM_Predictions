import pandas as pd
import struct
import openpyxl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.datasets import load_digits, fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
import sys
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import itertools
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn
import seaborn as sn



def main():
    #Import data
    data = pd.read_pickle("data.df")
    #Choose only selected features
    new_data = data[['FT%', '3P%', 'eFG%','ORB%', 'DRB%', 'AST%','TOV%', 'STL%', 'BLK%','USG%', 'PTS', 'PF','MP_per_PF', 'FTA_per_FGA', 'MP_per_3PA',
                     'PTS_per_FGA', 'SHUTTLE_RUN', 'THREE_QUARTER_SPRINT', 'STANDING_VERTICAL','MAX_VERTICAL', 'BENCH_PRESS',
                     'BODY_FAT','HAND_LENGTH', 'HAND_WIDTH', 'HEIGHT','HEIGHT_W_SHOES', 'REACH', 'WEIGHT', 'WINGSPAN', 'WM']].copy()
    

    
    #Hot encode classes
    yHOT = []
    yDF = new_data[['WM']]
    for index, row in yDF.iterrows():
        #Bad WM
        if row['WM'] < 0:
            yHOT.append([1,0,0])
            continue
        #Neutral WM
        if row['WM'] >= 0 and row['WM'] <= 1.5:
            yHOT.append([0,1,0])
            continue
        #Good WM
        if row['WM'] > 1.5:
            yHOT.append([0,0,1])
            continue
        
    del new_data['WM']

    #Remove highly correlated data
    cor_matrix = new_data.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    print(to_drop)
    new_removals = len(to_drop)
    for i in to_drop:
        del new_data[i]
    
    xDF = new_data.copy()

    x = xDF.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    xDF2 = pd.DataFrame(x_scaled)

    num_rows = len(xDF.index)

    xLIST = xDF2.values.tolist()

    #Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(xLIST, yHOT, train_size=533, test_size=130, shuffle=True)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    feature_vector_length = 29 - new_removals
    num_classes = 3
    best_accuracy = 0
    best_var = 0
    #Run 20 iterations of the model
    for i in range(20):
        # Set the input shape
        input_shape = (feature_vector_length,)
        print("Model: " + str(i))

        # Create the model
        model = Sequential()
        #Add first hidden layer
        model.add(Dense((20), input_shape=input_shape,use_bias=True, kernel_initializer='lecun_normal', bias_initializer="truncated_normal", activation='selu'))
        #Add first dropout layer
        model.add(Dropout(0.2))
        #Add second hidden layer
        model.add(Dense((15), kernel_initializer='lecun_normal', activation='selu'))
        #Add second dropout layer
        model.add(Dropout(0.2))
        #Add third hidden layer
        model.add(Dense((10), kernel_initializer='lecun_normal', activation='selu'))
        #Add output layer
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='Ftrl', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, verbose=0, validation_split=0.25)


        # Test the model after training
        test_results = model.evaluate(X_test, y_test, verbose=0)
        if best_accuracy < test_results[1]:
            best_accuracy = test_results[1]
            best_var = i
            best_model = model

    print(best_accuracy)
    print(best_var)
    test_results = best_model.evaluate(X_test, y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

    predic = best_model.predict(X_test)

    
    b = np.zeros_like(predic)
    b[np.arange(len(predic)), predic.argmax(1)] = 1
    b
    print(b)

    int_array = b.astype(int)

    y_pred = []
    y_act = []

    #Create confusion matrix
    predictions = int_array.tolist()
    truths = y_test.tolist()

    for i in range(len(predictions)):
        if predictions[i] == [1,0,0]:
            y_pred.append("Bad")
        elif predictions[i] == [0,1,0]:
            y_pred.append("Neutral")
        elif predictions[i] == [0,0,1]:
            y_pred.append("Good")

        if truths[i] == [1,0,0]:
            y_act.append("Bad")
        elif truths[i] == [0,1,0]:
            y_act.append("Neutral")
        elif truths[i] == [0,0,1]:
            y_act.append("Good")

    cm = confusion_matrix(y_act, y_pred, labels=["Bad", "Neutral", "Good"])
    print(cm)

    df_cm = pd.DataFrame(cm, range(3), range(3))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    plt.show()
main()
