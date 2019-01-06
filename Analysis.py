# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:50:14 2018

@author: lenovo
"""

from keras.models import model_from_json
import os
import cv2
import glob
import h5py
import pandas as pd
from sklearn.metrics import mean_absolute_error
import scipy.io as io
from PIL import Image
import numpy as np

def load_model():
    
    json_file = open('models/Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights/model_A_weights.h5")
    return loaded_model


def create_img(path):
    im = Image.open(path).convert('RGB')
    
    im = np.array(im)
    
    im = im/255.0
    
    im[:,:,0]=(im[:,:,0]-0.485)/0.229
    im[:,:,1]=(im[:,:,1]-0.456)/0.224
    im[:,:,2]=(im[:,:,2]-0.406)/0.225


    im = np.expand_dims(im,axis  = 0)
    return im

root = 'data'
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(len(img_paths))

model = load_model()
name = []
y_true = []
y_pred = []

for image in img_paths:
    name.append(image)
    gt = h5py.File(image.replace('.jpg','.h5').replace('images','ground') )
    groundtruth = np.asarray(gt['density'])
    num1 = np.sum(groundtruth)
    y_true.append(np.sum(num1))
    img = create_img(image)
    num = np.sum(model.predict(img))
    y_pred.append(np.sum(num))

    
data = pd.DataFrame({'name': name,'y_pred': y_pred,'y_true': y_true})
data.to_csv('CSV/A_on_A_test.csv', sep=',')

data = pd.read_csv('CSV/A_on_A_test.csv')
y_true = data['y_true']
y_pred = data['y_pred']

ans = mean_absolute_error(np.array(y_true),np.array(y_pred))

print("MAE : " , ans )

data = pd.read_csv('CSV/B_on_B_test.csv' , sep='\t')
y_true = data['y_true']
y_pred = data['y_pred']

ans = mean_absolute_error(np.array(y_true),np.array(y_pred))

print("MAE : " , ans )