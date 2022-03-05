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
# print(loss)
path = "D:/Databases/1)Training/UBFC-PHYS/s1/s1_t1_0_dataFile.hdf5"
#path = "D:/Databases/1)Training/COHFACE/1/0/data_dataFile.hdf5"
hf = h5py.File(path, 'r')
pulse = np.array(hf['pulse'])
peaks = np.array(hf['peaklist'])
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

temp = np.zeros(pulse.shape[0], dtype=np.float32)
m = 1/35
for i in range(0, len(peaks)):
    peak_1 = peaks[i]
    if(i-1 >= 0):
        peak_0 = peaks[i-1]
        min = round((peak_1 - peak_0)/2) + peak_0 + 1
        for j in range(min, peak_1+1):
            temp[j] = m*(peak_1 - j)
    elif(i-1 == -1):
        min = 0
        for j in range(min, peak_1+1):
            temp[j] = m*(peak_1 - j)
    if(i+1 < len(peaks)):
        peak_2 = peaks[i+1]
        max = round((peak_2 - peak_1)/2) + peak_1
        for j in range(peak_1, max+1):
            temp[j] = m*(j-peak_1)
    elif(i+1 == len(peaks)):
        max = len(temp)-1
        for j in range(peak_1, max+1):
            temp[j] = m*(j-peak_1)

model_output = [29, 65, 91] #[17, 32, 44] #= 
binary = np.zeros(pulse.shape[0], dtype=np.float32)
for index in model_output:
    binary[index] = 1

mult = binary * temp
mult2 = mult/0.1
sum = np.sum(mult)
sum2 = np.sum(mult2)

print(sum * 1/20)


plt.plot(temp)
plt.plot(binary)
plt.show()

plt.plot(mult)
plt.show()

