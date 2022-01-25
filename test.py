import glob
from typing import List
import cv2
import h5py
import scipy
from scipy.signal.filter_design import butter
from scipy.signal.signaltools import medfilt
from scipy.sparse.construct import spdiags
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from hrvanalysis import get_time_domain_features
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import heartpy  as hp




def find_csv(video_path):
    mms = MinMaxScaler()

    #f2 = h5py.File(video_path, "r")
    f2 = pd.read_csv(video_path)
    f3 = np.array(f2).flatten()
    #f3 = np.array(f2["pulse"][20:])
    f3_reshaped = f3.reshape(-1,1)
    
    peaks, peaks_ = signal.find_peaks(f3,distance=650*64/1000, height=20)# ,prominence=250)
    #print(peaks_)
    left = peaks[0] - 15
    right = peaks[0] + 25
    #print(left)
    #print(right)
    firstpeak = f3[left:right]
    #plt.plot(firstpeak)

    # fs = 64 #Hz
    # pulse_pred = (f3 - f3.min())/(f3.max() - f3.min()) * 2 -1
    # pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    # [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    # pulse_pred = signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    #print(firstpeak)
    #f4 = signal.filtfilt(firstpeak, len(firstpeak)-1)
    #b, a = signal.butter(3, 0.05)
    # #f4 = signal.filtfilt(b, a, firstpeak)
    # plt.plot(pulse_pred)
    # plt.show()
    # f3[left:right] = f4
    plt.plot(f3)
    plt.plot(peaks, f3[peaks], "x")
    
    f3= np.array(mms.fit_transform(f3_reshaped)).flatten()
    
    plt.show()

    
    nframe_per_video = 1304

    #f4 = f2["time"]
    #print(f2.keys())
    #print(f2.items())
    fs = 35.14
    data_resampled = np.array(signal.resample(f3, nframe_per_video ))
    
    
    ##### x Axis ####
    x_truth = np.arange(0, len(f3))
    x_resampled = np.linspace(0, len(f3), nframe_per_video)

    #### peaks #####
    peaks_truth, peaks_ = np.array(signal.find_peaks(f3, 0.4, distance=(550*(64/1000))))
    peaks_resampled, b  = np.array(signal.find_peaks(data_resampled, 0.4, distance=(550*(fs/1000))))
    #peaks_resamplednew = np.round(peaks_resampled*(fs/20))

    ###### HRV Features #######
    # truth_data, measures = hp.process(f3, 256, bpmmin=10, bpmmax=45)
    # resampled_data, measures_res = hp.process(data_resampled, 20, bpmmin=10, bpmmax=45)
    ##### IBI #####
    ibi_truth = np.diff(peaks_truth)*(1000/64)
    print(ibi_truth)
    ibi_res = np.diff(peaks_resampled)*(1000/fs)
    print(ibi_res)

    time_domain_features = get_time_domain_features(ibi_truth)
    time_domain_features_res = get_time_domain_features(ibi_res)
    print(time_domain_features)
    print(time_domain_features_res)


    print(len(ibi_truth))
    print(len(ibi_res))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x_truth, f3)
    plt.plot(peaks_truth, f3[peaks_truth], "x")
    
    plt.plot(x_resampled, data_resampled)
    plt.subplot(2,1,2)
    plt.plot(data_resampled)
    plt.plot(peaks_resampled, data_resampled[peaks_resampled], "x")
    # plt.subplot(2,1,2)
    # plt.plot(x_truth, ibi_truth)
    # plt.plot(x_resampled, ibi_res)
    
    plt.show()


    # hp.plotter(truth_data, measures, show = True)
    # hp.plotter(resampled_data, measures_res, show = True)
    
    # print(np.diff(peaks_truth))
    # print(np.diff(peaks_resampled))

    # average_pulse = np.sum(np.diff(peaks_truth))/(len(peaks_truth))
    # average_pulse_res = np.sum(np.diff(peaks_resampled))/(len(peaks_resampled))
    # print("HR: ", average_pulse*(1/256)*1000)
    # print("HR resampled: ", average_pulse_res*0.05*1000 )

    # print("f3", f3)
    # print("resampled", data_resampled)

#find_csv(video_path2)

def test_heartpy(path):
    mms = MinMaxScaler()

    f2 = h5py.File(path, "r")
    hrdata = np.array(f2["pulse"])
    #hrdata = hp.enhance_peaks(hrdata)

    mean = hrdata.mean()
    print(mean)
    print(hrdata.var())
    std = np.std(hrdata)
    print(std)
    
    upper_limit = mean + std*3
    lower_limit = mean - std*3

    for x in range(0, len(hrdata)):
        if hrdata[x] > upper_limit:
            hrdata[x] = upper_limit
        elif hrdata[x] < lower_limit:
            hrdata[x] = lower_limit
            

    #hrdata = hp.get_data(path)
    #hrdata_res = np.array(signal.resample(hrdata, 1700 ))
    hrdata = np.array(mms.fit_transform(hrdata.reshape(-1,1))).flatten()
    plt.plot(hrdata)
    plt.show()
    
    working_data, measures = hp.process(hrdata, 256, calc_freq=True)
    plot =  hp.plotter(working_data, measures, show=False)

    peak_list = working_data['peaklist'].tolist()
    print(type(peak_list))
    if not isinstance(peak_list, list):
        print("yes")
    else:
        print("no")
    removed = working_data['removed_beats']
    for item in removed:
        peak_list.remove(item)

    print(peak_list)

    plot.savefig("picture1.jpg")


    #print(working_data)
    print(measures)
    

#test_heartpy( "E:/Databases/Training/COHFACE/4/0/data.hdf5")

#databases = os.listdir(path , "1)Training")
#print(databases[0])

# print(os.getcwd())

# print("START!")
# list_gpu = tf.config.list_physical_devices('GPU')
# print("GPU:   ", list_gpu)
# tf.keras.backend.clear_session()
# print(tf.__version__)


# file = open("log.txt","w")
# file.write("Name:  ")
# file.write("fefefv\n" ), file.write("Train Subjects:  ")
# file.write("cscsc \n")
# file.close()