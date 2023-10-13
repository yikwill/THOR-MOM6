import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from numba.core.errors import NumbaDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

import shap
import numpy as np
import pylab as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import *
import matplotlib
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.io import loadmat
from pickle import load

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Input, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

scaler = load(open(f'Data_Step2/control/scaler.pkl', 'rb'))
print(scaler.mean_, scaler.scale_)

experiment = 'ssp585_2080_2099'
print(f'Experiment: {experiment}')
num_clusters = 6

missingdataindex = np.load(f'Data_Step2/{experiment}/missingdataindex.npy')
maskTraining = np.load(f'Data_Step2/{experiment}/maskTraining.npy')
maskVal = np.load(f'Data_Step2/{experiment}/maskVal.npy')
maskTest = np.load(f'Data_Step2/{experiment}/maskTest.npy')

total_features = np.load(f'Data_Step2/{experiment}/total_features.npy')
train_features = np.load(f'Data_Step2/{experiment}/train_features.npy')
val_features = np.load(f'Data_Step2/{experiment}/val_features.npy')
test_features = np.load(f'Data_Step2/{experiment}/test_features.npy')

X_total_scaled = scaler.transform(total_features)
X_train_scaled = scaler.transform(train_features)
X_val_scaled = scaler.transform(val_features)
X_test_scaled = scaler.transform(test_features)

print(X_train_scaled.shape)

feature_names = ['curlTau', 'col_height', 'zos', 'f', 'gradColHeight_x', 'gradColHeight_y', 'gradZos_x', 'gradZos_y', 'umo_2d', 'vmo_2d']

# For running SHAP on an ensemble of models
model_name = f'model_24x2_16x2_tanh_{num_clusters}_clusters'

if not os.path.isdir(f'shap_results/{model_name}/{experiment}'):
    os.makedirs(f'shap_results/{model_name}/{experiment}')
    print('Model directory created')
else:
    print('Model directory already exists')

shap_info_list = []

for i in range(50):
    tf.keras.backend.clear_session()
    model = None
    
    model = tf.keras.models.load_model(f'saved_models/{model_name}/model_{i}.h5')
    
    explainer = None
    explainer = shap.Explainer(model, X_train_scaled, algorithm='exact', feature_names=feature_names)
    shap_info = explainer(X_total_scaled)
    shap_info_list.append(shap_info)
    
    if i % 10 == 9:
        shap_vals = np.array([shap_info_list[i].values for i in range(len(shap_info_list))])
        shap_base_vals = np.array([shap_info_list[i].base_values for i in range(len(shap_info_list))])
        shap_data = np.array([shap_info_list[i].data for i in range(len(shap_info_list))])
        
        np.save(f'shap_results/{model_name}/{experiment}/shap_vals_{i+1}.npy', shap_vals)
        np.save(f'shap_results/{model_name}/{experiment}/shap_base_vals_{i+1}.npy', shap_base_vals)
        np.save(f'shap_results/{model_name}/{experiment}/shap_data_{i+1}.npy', shap_data)