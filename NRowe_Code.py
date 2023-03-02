#!/usr/bin/env python
# coding: utf-8

## IMPORTANT ##
'''
code was initally written in jupyter notebooks, so some of the print statements and
the final plot at the end may be formatted weird and not work. If there are any
major issues, please feel free to reach out or to put the code in a jupyter
notebook and run from there. 
'''

# # Import required packages and the data

# In[ ]:


# Normal Python Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Keras and sklearn NN imports
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import mean_squared_error

# load dataset
data = pd.read_pickle("data.df")


# # Scale and segregate the data

# In[2]:


# Get the columns that we care about
x_cols = ['gamesPlayed', 'minutes', 'FT%', '3P%', 'SOS',
        'PER', 'eFG%', 'ORB%', 'DRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%',
        'OWS', 'DWS', 'FTA', 'FGA', 'MP', '3PA', 'PTS', 'PF', 'MP_per_PF',
        'FTA_per_FGA', 'MP_per_3PA', 'PTS_per_FGA', 'ORtg', 'DRtg', "WING_DIFF",
        'awards', 'RSCI', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'SHUTTLE_RUN',
        'THREE_QUARTER_SPRINT', 'STANDING_VERTICAL', 'MAX_VERTICAL',
        'BENCH_PRESS', 'BODY_FAT', 'HAND_LENGTH', 'HAND_WIDTH', "didCombine", 
        'HEIGHT_W_SHOES', 'REACH', 'WEIGHT', 'WINGSPAN']

# Scale the columns using the standard scaler
for col in x_cols: 
    data.loc[:, col] = StandardScaler().fit_transform(data[col].to_numpy().reshape(-1, 1)) # Best one
    # data.loc[:, col] = MinMaxScaler().fit_transform(data[col].to_numpy().reshape(-1, 1))
    # data.loc[:, col] = Normalizer().fit_transform(data[col].to_numpy().reshape(-1, 1))
    pass
TARGET = "WM" # Delcare a target variable

# Split the dataset according to position
cData = data[data["C"]==1]
fData = data[data["F"]==1]
gData = data[data["G"]==1]


# In[3]:


optimizer =  "Adam" #"Adamax"# "SGD" # # "RMSprop"
loss = "mean_squared_error" # "mean_squared_logarithmic_error" # "mean_absolute_percentage_error" # "mean_absolute_error"
truth, predictions = [], []

models = 0


# # Center Model:

# In[4]:


# Build the center model
batch_size = 20
epochs = 20
f = 0.2

def c_model():
    # create model
    model = Sequential()
    model.add(Dense(N, input_dim=N, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(f))
    model.add(Dense(5, activation="linear"))
    model.add(Dense(1, kernel_initializer='normal', activation='linear' ))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
tData = cData.copy() # Get the right data

# Get the feature vector
X = tData[x_cols].copy()
X = X.astype(float)
N = len(x_cols)
# Get the target value
Y = tData[TARGET].copy()

mses = []
kfold = KFold(n_splits=5)
for train, test in kfold.split(tData):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = Y.iloc[train], Y.iloc[test]
    
    model = KerasRegressor(build_fn=c_model, epochs=epochs, batch_size=batch_size, verbose=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mses.append(mean_squared_error(preds, y_test))
    
    truth.extend(y_test)
    predictions.extend(preds)
    
print("Error: {:.2f}".format(np.mean(mses)))


# # Forward Model

# In[5]:


# Build the forward model
batch_size = 20
epochs = 20
f = 0.2
def f_model():
    # create model
    model = Sequential()
    model.add(Dense(N, input_dim=N, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(f))
    model.add(Dense(5, activation="linear"))
    model.add(Dense(1, kernel_initializer='normal', activation='linear' ))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
tData = fData.copy()

# Get the feature vector
X = tData[x_cols].copy()
X = X.astype(float)
N = len(x_cols)
# Get the target value
Y = tData[TARGET].copy()

mses = []
kfold = KFold(n_splits=5)
for train, test in kfold.split(tData):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = Y.iloc[train], Y.iloc[test]
    
    model = KerasRegressor(build_fn=f_model, epochs=epochs, batch_size=batch_size, verbose=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mses.append(mean_squared_error(preds, y_test))

    truth.extend(y_test)
    predictions.extend(preds)
    
print("Error: {:.2f}".format(np.mean(mses)))


# # Guard Model

# In[6]:


# Build the guard model
batch_size = 20
epochs = 20
f = 0.3
def g_model():
    # create model
    model = Sequential()
    model.add(Dense(N, input_dim=N, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(f))
    model.add(Dense(5, activation="linear"))
    model.add(Dense(1, kernel_initializer='normal', activation='linear' ))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
tData = gData.copy()

# Get the feature vector
X = tData[x_cols].copy()
X = X.astype(float)
N = len(x_cols)
# Get the target value
Y = tData[TARGET].copy()

mses = []
kfold = KFold(n_splits=5)
for train, test in kfold.split(tData):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = Y.iloc[train], Y.iloc[test]
    
    model = KerasRegressor(build_fn=g_model, epochs=epochs, batch_size=batch_size, verbose=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mses.append(mean_squared_error(preds, y_test))
    
    truth.extend(y_test)
    predictions.extend(preds)
    
print("Error: {:.2f}".format(np.mean(mses)))


# In[7]:


# See the overall error
print("Total Error: {:.2f}".format(mean_squared_error(truth, predictions)))


# In[8]:


plt.figure(figsize=(10,8), dpi=100)
m, b = np.polyfit(truth, predictions, 1)

plt.plot(truth, predictions, "o")
plt.plot(truth, m*np.array(truth) + b, '-')
plt.xlabel("Measued Value")
plt.ylabel("Predicted Value")
plt.grid()


# In[ ]:




