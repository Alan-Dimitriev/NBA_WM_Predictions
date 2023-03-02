from keras.layers import Layer
from keras import backend as K
import pandas as pd
import numpy as np
from keras.layers import Dense, Flatten, Input
from keras.models import Sequential
from keras.losses import binary_crossentropy
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import os 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    # defining weights of layer
    def build(self, input_shape):
        print(input_shape)
        print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    # logic for executing layer
    # basically the activation function for the RBF node
    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    # specifies output shape given input shape
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

def variance_selector(data, threshold=0.01):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_pickle(path+'/data.df')
print('Initial Dataset')
print(df.shape)
print(df.head())

# # Removing rows with null values for NBA team
# print('\nRemoving non NBA players')
# df = df[df.NBAteam.notnull()]
# print(df.shape)

# Filtering for features we want off the bat
print('\nChoosing features')
features = [
    'FT%', '3P%', 'eFG%', 'ORB%', 'DRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%', 'PTS', 'PF', 'MP_per_PF', 'FTA_per_FGA', 
    'MP_per_3PA', 'PTS_per_FGA', 'age', 'AGILITY', 'SHUTTLE_RUN', 'THREE_QUARTER_SPRINT', 'STANDING_VERTICAL', 'MAX_VERTICAL', 
    'BENCH_PRESS', 'BODY_FAT', 'HAND_LENGTH', 'HAND_WIDTH', 'HEIGHT', 'HEIGHT_W_SHOES', 'REACH', 'WEIGHT', 'WINGSPAN', 'WM'
]
df = df.filter(features)
print(df.shape)

print('\nRemoving rows with null')
df = df.dropna(how='any',axis=0) 
print(df.shape)

# defining features and labels
X = df.drop(labels=['WM'], axis=1)
y = df['WM']

# split training and testing
print('\nSplitting into training and test sets')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
print(X_train.shape, X_test.shape)

# remove constant features
print('\nRemoving Constant Features')
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X_train)

constant_columns = [column for column in X_train.columns
                    if column not in X_train.columns[constant_filter.get_support()]]

X_train.drop(labels=constant_columns, axis=1, inplace=True)
X_test.drop(labels=constant_columns, axis=1, inplace=True)
print(X_train.shape, X_test.shape)

# remove features under variance threshold
print('\nRemoving features under variance threshold')
X_train = variance_selector(X_train)
X_test = variance_selector(X_test)
print(X_train.shape, X_test.shape)
print(X_train.head())

# Filter out linearly correlated features
print('\nRemoving Linearly Correlated Features')
correlated_features = set()
corr = X.corr()
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
# plt.show()
for i in range(len(corr .columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.8:
            colname = corr.columns[i]
            correlated_features.add(colname)
X_train.drop(labels=correlated_features, axis=1, inplace=True)
X_test.drop(labels=correlated_features, axis=1, inplace=True)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Normalize data to between 0-1
print('\nNormalizing Data')
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.fit_transform(X_test)
y_train_norm = scaler.fit_transform(y_train.values.reshape(-1,1))
y_test_norm = scaler.fit_transform(y_test.values.reshape(-1,1))
print(X_train_norm.shape, X_test_norm.shape, y_train_norm.shape, y_test_norm.shape)

# define neural network
model = Sequential()
model.add(Input(shape=(X_train_norm.shape[1],)))
model.add(RBFLayer(530, 0.3))
model.add(Dense(1))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
# fit the keras model on the dataset
model.fit(X_train_norm, y_train_norm, epochs=100, batch_size=64)

# display graph comparing actual wins made vs predicted wins made
y_pred = model.predict(X_test_norm)
y_pred_train = model.predict(X_train_norm)
ax = sns.regplot(x=y_test_norm, y=y_pred)
ax.set(xlabel='measured', ylabel='predicted')
plt.show()

# evaluate the keras model
MSE = mean_squared_error(y_test_norm, y_pred)
MSE_training = mean_squared_error(y_train_norm, y_pred_train)
MSE_rescaled = np.sqrt(mean_squared_error(y_test, scaler.inverse_transform(y_pred)))
MSE_train_rescaled = np.sqrt(mean_squared_error(y_train, scaler.inverse_transform(y_pred_train)))
print('\nMSE Training:', MSE_training)
print('MSE Training Rescaled:', MSE_train_rescaled)
print('MSE Testing:', MSE)
print('MSE Testing Rescaled:', MSE_rescaled)


