import os
import scipy
from model import CAN_3D, PPTS_CAN, PTS_CAN, TS_CAN
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import heartpy as hp
import numpy as np
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend
from sklearn.preprocessing import MinMaxScaler



def gaussian_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, (-1,))
    return -tf.reduce_sum(tf.abs(y_true*y_pred))

path_of_video_tr = ["D:/Databases/3)Testing/COHFACE/25/1/data_dataFile.hdf5"]

model = PTS_CAN(10, 32, 64, (36,36,3),
                           dropout_rate1=0.25, dropout_rate2=0.5, nb_dense=128) #, parameter=['bpm', 'sdnn', 'pnn50', 'lf_hf']
training_generator = DataGenerator(path_of_video_tr, 2100, (36, 36),
                                           batch_size=1, frame_depth=10,
                                           temporal="PTS_CAN", respiration=False, database_name="COHFACE", 
                                           time_error_loss=True) # truth_parameter=['bpm', 'sdnn', 'pnn50', 'lf_hf']

inp = model.input   # input placeholder
#model.summary()
model_checkpoint = os.path.join("D:/Databases/4)Results/Version5/TS_CAN/cv_0_epoch24_model.hdf5")#PPTS_CAN_bpm_sdnn/cv_0_epoch24_model.hdf5")
model.load_weights(model_checkpoint)
#outputs = [layer.output for layer in model.layers]          # all layer outputs
#functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

# Testing
test = training_generator.data_generation(path_of_video_tr) 
output = model(test)

#### test new output ######
# pred = tf.reshape(output[2], (-1))
# truth = tf.reshape(test[1][2], (-1))
# AE = tf.abs(pred - truth)/truth
# loss = tf.reduce_mean(AE)
# diff = pred - truth
#print(output[1])
fs = 20
mms = MinMaxScaler()
pulse_pred = np.array(tf.reshape(output[0], (-1,)))
pulse_pred2 = detrend(np.cumsum(pulse_pred), 100)
[b_pulse_pred, a_pulse_pred] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
pulse_pred2 = scipy.signal.filtfilt(b_pulse_pred, a_pulse_pred, np.double(pulse_pred2))
pulse_pred2 = np.array(mms.fit_transform(pulse_pred2.reshape(-1,1))).flatten()

pulse_true = test[1][0]
working_data_pred, measures_pred= hp.process(pulse_pred, fs, calc_freq=True)
working_data_true, measures_true = hp.process(pulse_true, fs, calc_freq=True)
peaks_pred = working_data_pred['peaklist']
peaks_true = working_data_true['peaklist']
bin_arr = np.array(tf.reshape(output[1], (-1,)))
x_loss = np.where(bin_arr == 1)[0]

mult = np.array(tf.reshape(output[1], (-1,))) * test[1][1]
y_loss =np.delete(mult, np.where(bin_arr ==0))
plt.figure()
plt.subplot(211)
plt.title('TE loss function example')
plt.plot(pulse_pred, label='rPPG$_{out}$', linewidth=1, color="#B90F22")
plt.plot(peaks_pred, pulse_pred[peaks_pred], "x", color="#B90F22")
plt.plot(pulse_true, label='ground truth',linewidth=1,  color="#004E8A")
plt.plot(peaks_true, pulse_true[peaks_true], "x", color="#004E8A")
plt.ylabel("rPPG [a.u.]")
plt.legend(loc="upper right")
plt.subplot(212)
plt.title('Corrensponding Loss')
plt.vlines(np.where(output[1] == 1), ymin=0, ymax=1 , label='binary$_{out}$', color="#B90F22")#
plt.hlines(0, 0, 505, color= "#B90F22")
plt.plot(test[1][1], label='ground truth',linewidth=1, color="#004E8A")
plt.plot(x_loss, y_loss , "x", color="k", label="Loss")
plt.ylabel("seconds")
plt.xlabel("time (samples)")
plt.legend(loc="upper right")
plt.show()


y_true = test[1][1]
y_pred = output[1]
loss = gaussian_loss(y_true, y_pred)
mult = y_true * tf.reshape(y_pred, (-1,))
print(np.sum(mult))
plt.plot(mult)
plt.title("Multiplication of y_true and y_pred")
plt.ylabel("[a.u.]")
plt.xlabel("time (samples)")
plt.show()

#layer_outs = [func([test, 1.]) for func in functors]
#print(layer_outs)


