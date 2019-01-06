# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:32:23 2018
生成groundh5文件
# Data Preprocessing  :
In data preprocessing, the main objective was to convert the ground truth 
provided by the ShanghaiTech dataset into density maps. For a given image the dataset 
provided a sparse matrix consisting of the head annotations in that image. 
This sparse matrix was converted into a 2D density map by passing through a Gaussian Filter.
The sum of all the cells in the density map results in the actual count of people in that 
particular image. Refer the `Preprocess.ipynb` notebook for the same.

# Data preprocessing math explained:
Given a set of head annotations our task is to convert it to a density map.
1) Build a kdtree(a kdtree is a data structure that allows fast computation of K 
Nearest neighbours) of the head annotations.
2) Find the average distances for each head with K(in this case 4) nearest heads in the head
 annotations. Multpiply this value by a    factor, 0.3 as suggested by the author of the paper.
3) Put this value as sigma and convolve using the 2D Gaussian filter. 

@author: lenovo
"""

import h5py
import scipy.io as io
import PIL as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
from matplotlib import cm as CM
from image import *

def gaussian_filter_density(gt):
    #Generates a density map using Gaussian filter transformation   
    density = np.zeros(gt.shape, dtype=np.float32)
    
    gt_count = np.count_nonzero(gt)
    
    if gt_count == 0:
        return density
    # FInd out the K nearest neighbours using a KDTree
   
    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048
    
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    
    # query kdtree
    distances, locations = tree.query(pts, k=4)
       
    for i, pt in enumerate(pts):#枚举，返回序列（索引，数据）
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point        
        #Convolve with the gaussian filter
        
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')    
    return density

root = 'data'
#part_A_train = os.path.join(root,'part_A_final/train_data','images')
#part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_B_train,part_B_test]

# List of all image paths

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(len(img_paths))

from tqdm import tqdm
i = 0
for img_path in tqdm(img_paths):#tqdm显示进度条
           
    # Load sparse matrix
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    
    #Read image
    img= plt.imread(img_path)
    
    # Create a zero matrix of image size
    k = np.zeros((img.shape[0],img.shape[1]))
    
    gt = mat["image_info"][0,0][0,0][0]
    
    #Generate hot encoded matrix of sparse matrix
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    
    # generate density map
    k = gaussian_filter_density(k)
    
    # File path to save density map
    file_path = img_path.replace('.jpg','.h5').replace('images','ground')
    
    #break
    with h5py.File(file_path, 'w') as hf:#打开文件并最后关闭
            hf['density'] = k
            
file_path = img_paths[22].replace('.jpg','.h5').replace('images','ground') 
print(file_path)
#Sample Ground Truth
gt_file = h5py.File(file_path,'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)#CM.jet显示蓝-青-黄-红四色
print("Sum = " ,np.sum(groundtruth))