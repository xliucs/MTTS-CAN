import code
from dataclasses import replace
import math
from cv2 import threshold
import cv2
import numpy as np
import tensorflow as tf
import h5py

import heartpy as hp
import matplotlib.pyplot as plt
# def get_peaks(y):
#         # y: (N,1)
#         data_reshaped = tf.reshape(y, (1, -1, 1)) # (1, N, 1)
#         max_pooled_in_tensor =  tf.nn.max_pool(data_reshaped, (20,), 1,'SAME')
#         maxima = tf.equal(data_reshaped,max_pooled_in_tensor) # (1, N, 1)
#         maxima = tf.cast(maxima, tf.float32)
#         maxima = tf.squeeze(maxima) # (N,1)
#         maxima = tf.reshape(maxima, (-1,1))
#         #peaks = tf.where(maxima) # now only the Peak Indices (A, 3)
#         #
#         # peaks = tf.reshape(peaks, (-1,)) # (A,1)

#         return maxima
# print(get_peaks(1))

# def gauss(x, sigma, mu):
#     return math.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))


# a = [2,28,132,163,174,196, 48,70,91,112]
# b = [50]#[3,25, 50, 77, 91]

# temp = np.zeros(100)
# temp2 = np.zeros(100)

# for k in b:
#     temp2[k] = 1
# plt.plot(temp2)
# plt.show()

# for i in b:
#     mu = i
#     sigma = 3
#     for j in range(i-sigma*3, i+sigma*3):
#         temp[j] = gauss(j, sigma, mu)

# pred = np.zeros(100)
# pred[45] = 1

# plt.plot(temp)
# plt.plot(pred)
# plt.show()

# def gaussian_loss(y_true, y_pred):
#     return -tf.reduce_sum(y_true*y_pred)

# loss = gaussian_loss(temp,pred)

import pandas as pd
from importlib import import_module
import numpy as np
import cv2
from skimage.util import img_as_float
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import scipy.io
from scipy.sparse import spdiags
from tensorflow.python.keras import backend as K

def preprocess_raw_video(videoFilePath, dim=36):

    #########################################################################
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath)
    
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) # get total frame size
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    #print("fps:   ", fps)
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype = np.float32)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()
    dims = img.shape
    #print("Orignal Height", height)
    #print("Original width", width)
    
    #########################################################################
    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))# current timestamp in milisecond
        vidLxL = cv2.resize(img_as_float(img), (dim, dim), interpolation = cv2.INTER_AREA) #img[:, int(width/2)-int(height/2 + 1): int(height/2)+int(width/2), :])
        #vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE) # rotate 90 degree
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1/255)] = 1/255
        Xsub[i, :, :, :] = vidLxL
        success, img = vidObj.read() # read the next one
        i = i + 1
    # plt.imshow(Xsub[0])
    # plt.title('Sample Preprocessed Frame')
    # plt.show()
    #########################################################################
    # Normalized Frames in the motion branch
    normalized_len = len(t) - 1

    #print("normalized Len")
    #print(normalized_len)
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype = np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])
    dXsub = dXsub / np.std(dXsub)
    # plt.imshow(dXsub[0])
    # plt.title('Sample Preprocessed Frame')
    # plt.show()

    #########################################################################
    # Normalize raw frames in the apperance branch
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub  / np.std(Xsub)
    Xsub = Xsub[:dXsub.shape[0], :, :, :] # -1
    #########################################################################
    # Plot an example of data after preprocess
    dXsub = np.concatenate((dXsub, Xsub), axis = 3)
    return dXsub, fps

import ast
# print(loss)
subTrain = np.array(range(1,34))
subTest = np.array([range(34,42)])
print(subTrain)
print("JKBJKBJB:",subTest)
#path = "D:/Databases/1)Training/COHFACE/1/0/data_dataFile.hdf5"
#path = "D:/Databases/1)Training/UBFC-PHYS/s6/s6_T1_dataFile.hdf5"
#vid = "D:/Databases/1)Training/UBFC-PHYS/s6/vid_s6_T1.avi"

path = "D:/Databases/UBFC/subject33/ground_truth.txt"
test = open(path, 'r').read()
test = str(test).split("  ")
test = test[1:]
print(len(test))
results = list(map(float, test[0: int(len(test)/3)]))

plt.plot(results)
plt.show()

#dxSub = preprocess_raw_video(vid, 36)
#path = "D:/Databases/1)Training/COHFACE/1/0/data_dataFile.hdf5"
hf = h5py.File(path, 'r')
vidData = hf['parameter']
numer = np.array(vidData)
test2 = ast.literal_eval(str(numer))
bpm = test2["bpm"]
fs = 35

peak_truth = np.array(hf['peaklist'])
ibi_arr = []
for peak in range(0,len(peak_truth)-1):
    ibi = ((peak_truth[peak+1] - peak_truth[peak])/fs)*1000
    print(ibi)
    if ibi<1500 and ibi > 333:
        ibi_arr.append(ibi)

mean = np.mean(ibi_arr)
HR = 60000/mean
diff = ibi_arr - mean
test = diff **2
sdnn = np.sqrt(1/(len(diff)-1)*np.sum(diff**2))

print(ibi_arr)
pulseDiscrete = np.array(hf['pulse'])
#peaks = np.array(hf['peaklist'])
hf = pd.read_csv(path)
pulse = np.array(hf)

plt.plot(pulse)
plt.show()

#working_data, measures = hp.process(pulse.reshape(-1,), 35.14, calc_freq=True)
#plot =  hp.plotter(working_data, measures, show=False, title = 'Heart Rate Signal and Peak Detection')
#plt.show()

# y: (N,1)
# data_reshaped = tf.reshape(pulse, (1, -1, 1)) # (1, N, 1)
# max_pooled_in_tensor = tf.nn.max_pool(data_reshaped, (20,), 1,'SAME')
# maxima = tf.equal(data_reshaped, max_pooled_in_tensor) # (1, N, 1)
# maxima = tf.cast(maxima, tf.float32)
# #maxima = tf.squeeze(maxima) # (N,1)
# maxima = tf.reshape(maxima, (-1,))
# peaks = tf.where(maxima) # now only the Peak Indices (A, 3)
# #
# peaks = tf.reshape(peaks, (-1,)) # (A,1)

# plt.plot(maxima)
# plt.title("Binary Peak signal")
# plt.xlabel("time (sample)")
# plt.ylabel("normalized signal [a. u.]")
# plt.show()

# temp = np.zeros(pulse.shape[0], dtype=np.float32)
# m = 1/35
# for i in range(0, len(peaks)):
#     peak_1 = peaks[i]
#     if(i-1 >= 0):
#         peak_0 = peaks[i-1]
#         min = round((peak_1 - peak_0)/2) + peak_0 + 1
#         for j in range(min, peak_1+1):
#             temp[j] = m*(peak_1 - j)
#     elif(i-1 == -1):
#         min = 0
#         for j in range(min, peak_1+1):
#             temp[j] = m*(peak_1 - j)
#     if(i+1 < len(peaks)):
#         peak_2 = peaks[i+1]
#         max = round((peak_2 - peak_1)/2) + peak_1
#         for j in range(peak_1, max+1):
#             temp[j] = m*(j-peak_1)
#     elif(i+1 == len(peaks)):
#         max = len(temp)-1
#         for j in range(peak_1, max+1):
#             temp[j] = m*(j-peak_1)

# model_output = [29, 65, 91] #[17, 32, 44] #= 
# binary = np.zeros(pulse.shape[0], dtype=np.float32)
# for index in model_output:
#     binary[index] = 1

# mult = binary * temp
# sum = np.sum(mult)
# print(sum)


# plt.plot(temp)
# plt.plot(binary)
# plt.show()

# plt.plot(mult)
# plt.show()
