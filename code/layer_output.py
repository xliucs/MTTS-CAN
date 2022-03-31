

import os
from sklearn.preprocessing import MinMaxScaler
from model import CAN_3D, PPTS_CAN, PTS_CAN, TS_CAN
from tensorflow import keras
from keras import backend as K
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py



def gaussian_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, (-1,))
    return -tf.reduce_sum(tf.abs(y_true*y_pred))

path_of_video_tr = ["D:/Databases/1)Training/COHFACE/5/1/data_dataFile.hdf5"]#,"D:/Databases/1)Training/COHFACE/2/1/data_dataFile.hdf5"]
                  #  "D:/Databases/1)Training/COHFACE/1/2/data_dataFile.hdf5","D:/Databases/1)Training/COHFACE/1/3/data_dataFile.hdf5"]

model = PPTS_CAN(10, 32, 64, (36,36,3),
                           dropout_rate1=0.25, dropout_rate2=0.5, nb_dense=128, parameter=['bpm', 'sdnn', 'pnn50', 'lf_hf'])
training_generator = DataGenerator(path_of_video_tr, 2100, (36, 36),
                                           batch_size=1, frame_depth=10,
                                           temporal="PPTS_CAN", respiration=False, database_name="COHFACE", 
                                           time_error_loss=True, truth_parameter=['bpm', 'sdnn', 'pnn50', 'lf_hf'])

inp = model.input   # input placeholder
#model.summary()
model_checkpoint = os.path.join("D:/Databases/4)Results/altesPPTS/PPTS_CAN_bpm_sdnn/cv_0_epoch24_model.hdf5")
model.load_weights(model_checkpoint)
#outputs = [layer.output for layer in model.layers]          # all layer outputs
#functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

# Testing
test = training_generator.data_generation(path_of_video_tr) 
output = model(test)
# import heartpy
# heartpy.process
# import scipy.signal as sc

# from scipy.interpolate import UnivariateSpline
# ##################
# path = "D:/Databases/1)Training/COHFACE/6/1/data_dataFile.hdf5"
# h5pyfiledata = h5py.File(path, 'r')
# data = np.array(h5pyfiledata["pulse"])

# f,m = heartpy.process(data, 20, freq_method='fft',calc_freq=True)

# rr_list = f['RR_list_cor']
# # Aggregate RR-list and interpolate to a uniform sampling rate at 4x resolution
# rr_x = np.cumsum(rr_list)

# resamp_factor = 4
# datalen = int((len(rr_x) - 1)*resamp_factor)
# rr_x_new = np.linspace(int(rr_x[0]), int(rr_x[-1]), datalen)

# interpolation_func = UnivariateSpline(rr_x, rr_list, k=3)
# rr_interp = interpolation_func(rr_x_new)

# # RR-list in units of ms, with the sampling rate at 1 sample per beat
# dt = np.mean(rr_list) / 1000  # in sec
# fs = 1 / dt  # about 1.1 Hz; 50 BPM would be 0.83 Hz, just enough to get the
# # max of the HF band at 0.4 Hz according to Nyquist
# fs_new = fs * resamp_factor

# # compute PSD (one-sided, units of ms^2/Hz)
# frq = np.fft.fftfreq(datalen, d=(1 / fs_new))
# frq = frq[range(int(datalen / 2))]
# Y = np.fft.fft(rr_interp) / datalen
# Y = Y[range(int(datalen / 2))]

# plt.plot(frq,Y, label="X")
# plt.show()
# psd = np.power(Y, 2)

# df = frq[1] - frq[0]
# lf = np.trapz(abs(psd[(frq >= 0.04) & (frq < 0.15)]), dx=df)
# hf = np.trapz(abs(psd[(frq >= 0.15) & (frq < 0.4)]), dx=df)
# print("mit Interpolation")
# print(lf, m['lf'])
# print(hf, m['hf'])

# f, pxx = sc.periodogram(data, 20)

# f2,m2 = heartpy.process(data, 20,calc_freq=True)
# print(m2['lf'], m['lf'])
# print(m2['hf'], m['hf'])
# print(m2['lf/hf'], m['lf/hf'])

# #plt.plot(data)
# #plt.show()
# data = np.array(h5pyfiledata["nn"])
# size = tf.size(data)
# print("SIZE",size )
# b = tf.range(tf.cast(0, tf.float32), tf.cast(size, tf.float32), tf.cast(1/20, tf.float32))

# data = tf.reshape(tf.convert_to_tensor(data), (-1,))
# #data = tf.cast(data, dtype=tf.complex64)

# frq = tf.cast(tf.abs(tf.signal.rfft(data)), tf.float32)/tf.cast(size,tf.float32)#, tf.cast(size, tf.float32))#/size#, 60, 1200))

# dt = np.mean(data) / 1000  # in sec
# fs = 1 / dt
# t = np.fft.fftfreq(63, d=(1 / fs))
# t = t[0:tf.size(frq)]
# print(t)

# t = np.arange(0,(tf.size(frq)))
# t = tf.cast(tf.range(0, tf.size(frq)), tf.float32)
# print("NEU",t)
# t = tf.cast(t, tf.float32)/(tf.cast(dt, tf.float32)*tf.cast(tf.size(frq)*2, tf.float32))
# print(t)
# #frq = tf.slice(frq, 1,tf.size(data)/2 )

# plt.plot(t, frq, label="FFT")
# plt.show()

# mask_lf = tf.cast(tf.logical_and(tf.greater_equal(t, 0.04), tf.less(t, 0.15)), tf.float32)
# lf = tf.maximum(tf.reduce_sum(frq*mask_lf), 0.000001)
# mask_hf = tf.cast(tf.logical_and(tf.greater_equal(t, 0,15), tf.less(t, 0.4)), tf.float32)
# hf = tf.maximum(tf.reduce_sum(frq*mask_lf), 0.000001)

# lf_hf = lf/hf

# print(lf)


# test = tf.squeeze(frq)

# b = tf.abs(test**2)/(2*20)


# # fs = 20
# # indices = tf.where(tf.equal(tf.reshape(maxima, (-1,)),1))
# # print("pred", indices)
# # peak_locations = tf.squeeze(indices)

# def tf_diff_axis_0(a):
#     return a[1:]-a[:-1]
# ibi_arr = tf_diff_axis_0(data)*50

# #mask = tf.logical_and(tf.greater_equal(ibi_arr,333),tf.less_equal(ibi_arr, 1500))
# #mask.set_shape([None])

# #rr_arr = tf.boolean_mask(ibi_arr, mask)
# HR = 60000/tf.reduce_mean(ibi_arr)
# rr_mean = tf.math.reduce_std(tf.cast(ibi_arr, dtype=tf.float32))
# #HR = 60000/rr_mean#
# print("HR " ,rr_mean)

#### test new output ######
pred = tf.reshape(output[2], (-1))
truth = tf.reshape(test[1][2], (-1))
AE = tf.abs(pred - truth)/truth
loss = tf.reduce_mean(AE)
diff = pred - truth
#print(output[1])

plt.plot(output[0], label="Prediction")
plt.plot(test[1][0], label="truth")
plt.plot(maxima)
plt.legend()
plt.show()

# plt.figure()
# plt.subplot(211)
# plt.title('Predicted signal example')
# plt.plot(output[0][100:500], label='output 1')
# plt.ylabel("rPPG [a.u.]")
# plt.legend(loc="upper right")
# plt.subplot(212)
# plt.plot(output[1][100:500], label='output 2')
# #plt.ylabel("[a.u.]")
# plt.legend(loc="upper right")
# plt.show()

# plt.figure()
# plt.subplot(211)
# plt.title('Ground truth example')
# plt.plot(test[1][0][100:500], label='truth data 1')
# plt.ylabel("rPPG [a.u.]")
# plt.legend(loc="upper right")
# plt.subplot(212)
# plt.plot(test[1][1][100:500], label='truth data 2')
# plt.xlabel("time (samples)")
# plt.legend(loc="upper right")
# plt.show()

# y_true = test[1][1]
# y_pred = output[1]
# loss = gaussian_loss(y_true, y_pred)
# mult = y_true * tf.reshape(y_pred, (-1,))
# plt.plot(mult)
# plt.title("Multiplication of y_true and y_pred")
# plt.ylabel("[a.u.]")
# plt.xlabel("time (samples)")
# plt.show()

#layer_outs = [func([test, 1.]) for func in functors]
#print(layer_outs)


