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




#dxSub = preprocess_raw_video(vid, 36)
path = "D:/Databases/1)Training/COHFACE/2/0/data_dataFile.hdf5"
hf = h5py.File(path, 'r')

hr_data = np.array(hf['nn'])


hr_data = tf.convert_to_tensor(hr_data)

def tf_diff_axis_0(a):
    return a[1:]-a[:-1]
ibi_diff = tf_diff_axis_0(hr_data)

mask = (tf.greater(tf.abs(ibi_diff),50))
mask.set_shape([None])

mask = tf.cast(mask, dtype=tf.int32)
nn50 = tf.math.reduce_sum(mask)

rr_arr = tf.boolean_mask(ibi_diff, mask)

print(nn50/tf.size(mask))
print(np.array(hf['parameter']))




from scipy.interpolate import UnivariateSpline

working_data, measures = hp.process(hr_data, 20, calc_freq=True)

rr_list = np.array(hf['nn'])

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

# nperseg should be based on trade-off btw temporal res and freq res
# default is 4 min to get about 9 points in the VLF band
welch_wsize=240
nperseg = welch_wsize * fs_new
if nperseg >= len(rr_x_new):  # if nperseg is larger than the available data segment
    nperseg = len(rr_x_new)  # set it to length of data segment to prevent scipy warnings
    # as user is already informed through the signal length warning
frq, psd = ss.welch(rr_list, fs=fs)#, nperseg=nperseg)


plt.plot(frq,psd)
plt.show()

df = frq[1] - frq[0]

lf = 0
for fs in range(0, len(frq)):
    if(frq[fs]>= 0.04 and frq[fs]<0.15):
        lf += psd[fs]
print(lf)
print(lf*df)
lf= np.trapz(abs(psd[(frq >= 0.04) & (frq < 0.15)]), dx=df)
hf_val = np.trapz(abs(psd[(frq >= 0.15) & (frq < 0.4)]), dx=df)
lf_hf = lf /hf_val


print("HF", hf_val)
print("LF", lf)
print("LF/hf", lf_hf)
print("TRuth:  ", np.array(hf['parameter']))