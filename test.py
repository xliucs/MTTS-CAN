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
import scipy.signal as ss
import scipy.fft as sf


from scipy.interpolate import UnivariateSpline
##################
path = "D:/Databases/1)Training/COHFACE/6/1/data_dataFile.hdf5"
h5pyfiledata = h5py.File(path, 'r')
data = np.array(h5pyfiledata["pulse"])


rr_list = np.array(h5pyfiledata['nn'])
# Aggregate RR-list and interpolate to a uniform sampling rate at 4x resolution
rr_x = np.cumsum(rr_list)

resamp_factor = 4
datalen = int((len(rr_x) - 1)*resamp_factor)
rr_x_new = np.linspace(int(rr_x[0]), int(rr_x[-1]), datalen)

interpolation_func = UnivariateSpline(rr_x, rr_list, k=3)
rr_interp = interpolation_func(rr_x_new)


# RR-list in units of ms, with the sampling rate at 1 sample per beat
dt = np.mean(rr_list) / 1000  # in sec
fs = 1 / dt  # about 1.1 Hz; 50 BPM would be 0.83 Hz, just enough to get the
# max of the HF band at 0.4 Hz according to Nyquist
fs_new = fs * resamp_factor

# compute PSD (one-sided, units of ms^2/Hz)
frq = np.fft.fftfreq(datalen, d=(1 / fs_new))
frq = frq[range(int(datalen / 2))]
Y = np.math.sqrt(2)* np.fft.rfft(rr_interp) / datalen
Y = Y[:-1]
#Y = Y[range(int(datalen / 2))]
Y = np.power(Y,2)

delta = fs/tf.size(rr_list)

frq3 = tf.cast(tf.range(0, tf.size(frq)), tf.float32)
frq3 = tf.cast(frq3, tf.float32)/(tf.cast(dt, tf.float32)*tf.cast(tf.size(rr_list), tf.float32))
frq2 = np.fft.rfftfreq(len(rr_x), d=(1 / fs))
#frq2 = frq2[range(int(len(rr_x) / 2))]
Y2= np.math.sqrt(2)*  np.fft.rfft(rr_list) / len(rr_x) 


#
#Y2 = Y[range(int(len(rr_x) / 2))]
Y2 = np.power(Y2,2)

plt.figure()
plt.subplot(211)
plt.title("Raw RR intervall data")
plt.plot(rr_list, label="raw RR signal", color ='#005AA9')
plt.ylabel("ms")
plt.subplot(212)
plt.title("RR time-series with interpolation")
plt.plot(rr_interp, label="with interpolation", color ='#005AA9')
plt.ylabel("ms")

plt.xlabel("samples")
plt.show()


y = np.arange(-20,400,0.1)
plt.plot(frq,np.abs(Y), label="interpolated Power Spectrum", color ='#E6001A')
plt.ylabel("$ms^{2}$")
plt.xlabel("Hz")
plt.title("FFT Power Spectrum")
plt.fill_betweenx(y, 0.04, 0.15,  color='#7FAB16', alpha=.3)
plt.fill_betweenx(y, 0.15, 0.4,  color='#D28700', alpha=.3)
plt.plot(frq2,np.abs(Y2), label="raw Power Spectrum", color ='#005AA9')

plt.legend()
plt.show()
psd = np.power(Y, 2)

mask_lf = tf.cast(tf.logical_and(tf.greater_equal(frq, 0.04), tf.less(frq, 0.15)), tf.float32)
lf = tf.reduce_sum(np.abs(Y)*mask_lf)
mask_hf = tf.cast(tf.logical_and(tf.greater_equal(frq, 0.15), tf.less(frq, 0.4)), tf.float32)
hf = tf.reduce_sum(np.abs(Y)*mask_hf)

mask_lf2 = tf.cast(tf.logical_and(tf.greater_equal(frq2, 0.04), tf.less(frq2, 0.15)), tf.float32)
lf2 = tf.reduce_sum(np.abs(Y2)*mask_lf2)
mask_hf2 = tf.cast(tf.logical_and(tf.greater_equal(frq2, 0.15), tf.less(frq2, 0.4)), tf.float32)
hf2 = tf.reduce_sum(np.abs(Y2)*mask_hf2)

print("inter: ", lf, "not: ", lf2)
print("inter: ", hf, "not:", hf2)

import heartpy as hp

f, m = hp.process(data, 20, freq_method='fft', calc_freq=True)

print(m)