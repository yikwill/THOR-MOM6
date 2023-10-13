import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Input, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow import keras 

from pickle import dump

experiment = 'control'
print(f'Experiment: {experiment}')

missingdataindex = np.load(f'Data_Step2/{experiment}/missingdataindex.npy')
maskTraining = np.load(f'Data_Step2/{experiment}/maskTraining.npy')
maskVal = np.load(f'Data_Step2/{experiment}/maskVal.npy')
maskTest = np.load(f'Data_Step2/{experiment}/maskTest.npy')

num_clusters = 6
print(f'Predicting {num_clusters} clusters')

total_features = np.load(f'Data_Step2/{experiment}/total_features.npy')
total_labels = np.load(f'Data_Step2/{experiment}/total_labels_{num_clusters}_clusters.npy')

train_features = np.load(f'Data_Step2/{experiment}/train_features.npy')
train_labels = np.load(f'Data_Step2/{experiment}/train_labels_{num_clusters}_clusters.npy')

val_features = np.load(f'Data_Step2/{experiment}/val_features.npy')
val_labels = np.load(f'Data_Step2/{experiment}/val_labels_{num_clusters}_clusters.npy')

test_features = np.load(f'Data_Step2/{experiment}/test_features.npy')
test_labels = np.load(f'Data_Step2/{experiment}/test_labels_{num_clusters}_clusters.npy')

scaler = StandardScaler()
scaler.fit(train_features)
#scaler.mean_, scaler.scale_

X_total_scaled = scaler.transform(total_features)
Y_total = tf.keras.utils.to_categorical(total_labels)

X_train_scaled = scaler.transform(train_features)
Y_train = tf.keras.utils.to_categorical(train_labels)

X_val_scaled = scaler.transform(val_features)
Y_val = tf.keras.utils.to_categorical(val_labels)

X_test_scaled = scaler.transform(test_features)
Y_test = tf.keras.utils.to_categorical(test_labels)

print(X_train_scaled.shape, Y_train.shape)

model_name = f'model_24x2_16x2_tanh_{num_clusters}_clusters'

if not os.path.isdir(f'saved_models/{model_name}/'):
    os.makedirs(f'saved_models/{model_name}/')
    print('Model directory created')
else:
    print('Model directory already exists')
    
for i in range(50):
    
    print(f'Training model {i}')
    model = None
    tf.keras.backend.clear_session()
    member_name = f'model_{i}'

    inputs = Input(shape=(10,))
    x = Dense(24, activation='tanh')(inputs)
    x = Dense(24, activation='tanh')(x)
    x = Dense(16, activation='tanh')(x)
    x = Dense(16, activation='tanh')(x)
    outputs = Dense(num_clusters, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)])
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    #lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)
    
    history = model.fit(X_train_scaled, Y_train,
                        batch_size=32,
                        epochs=100,
                        verbose=0,
                        validation_data=(X_val_scaled, Y_val),
                        shuffle=True,
                        callbacks=[es])
    
    model.save(f'saved_models/{model_name}/{member_name}.h5')