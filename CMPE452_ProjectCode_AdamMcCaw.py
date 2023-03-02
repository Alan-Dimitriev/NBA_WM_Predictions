#!/usr/bin/env python
# coding: utf-8

# In[59]:


#4 Layer Regressive MLP to Predict NBA Wins Made
#Written by Adam McCaw
#Dec. 8, 2020

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow import keras
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data = pd.read_pickle("Sportsdata.df")
#define expected output as WM column
output = data["WM"]

#delete uneccesary columns in the dataset
del data["EWA"]
del data["WS"]
del data["WP"]
del data["WM"]
del data["did3years"]
del data["didAnyYears"]
del data["didCombine"]
del data["Player"]

#show dataset shape
print(data.shape)


# In[60]:


#normalize the input
data = np.array(data)
for i in range (len(data[0])):
    for j in range (len(data)):
        #standard normalization formula with (value - mean)/standard deviation for each feature
        data[j][i] = (data[j][i] - np.mean(data[:][i]))/(np.std(data[:][i]))
print(data)




# In[73]:



MSEsum = 0
MSEtrainSum = 0
MSEmin = 100
#run the model 50 times while shuffling the dataset each time
for i in range(50):  
    X_train, X_test, y_train, y_test = train_test_split(data, output, train_size = 0.85, test_size = 0.15, shuffle = True)

    #initialize the model
    model = Sequential()

    #Add the model layers with their node # and activation
    model.add(Dense(21, input_dim = 53, activation = 'sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(7, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))
    
    #compile the model defining loss function, optimizer, and metrics for determining success
    model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
    
    #fit the model to the training data with 100 epochs and 15% of the training data used as validation
    model.fit(X_train, y_train, epochs = 100, verbose = 0, validation_split = 0.15)
    
    #evaluate the model on the testing and training data to determine MSEs
    _, MSEtest = model.evaluate(X_test, y_test, verbose = 0)

    _, MSEtrain = model.evaluate(X_train, y_train, verbose = 0)

    MSEsum+=MSEtest
    MSEtrainSum += MSEtrain
    if (MSEmin > MSEtest):
        MSEmin = MSEtest
        bestPredict = model.predict(X_test).flatten()
    print("Model Test: #" + str(i+1))
    
MSEavg = round(MSEsum/50,2)
MSEmin = round(MSEmin,2)
MSETrainAvg = round(MSEtrainSum/50,2)
print("Average MSE:")
print(MSEavg)
print("Minimum MSE:")
print(MSEmin)
print("Average Training MSE:")
print(MSETrainAvg)

#plot the residuals for the minMSE result of the 50 model evaluations
residuals = np.subtract(bestPredict,y_test)
plt.figure(figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
x = np.linspace(0,100,100)
plt.plot(x,residuals, 'o' , label = "Data Residuals")
y = np.zeros(100)
plt.plot(x,y, label = "Perfect Result")
plt.title("Residual Plot of MSE Min Result")
plt.ylabel("Residuals (Prediction - Expected)")
plt.xlabel("Sample Number")
plt.legend()

