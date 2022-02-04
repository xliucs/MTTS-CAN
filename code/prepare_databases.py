from scipy.sparse import data
from pre_process import sort_video_list_, split_subj_
from inference_preprocess import preprocess_raw_video

import h5py
import itertools
import numpy as np
from scipy import signal
import os
from sklearn.preprocessing import MinMaxScaler
import heartpy as hp
import pandas as pd

def hr_analysis(path, hr_discrete, frames_vid, fps_vid):
    mms = MinMaxScaler()
    mean = hr_discrete.mean()
    std = np.std(hr_discrete)
    
    upper_limit = mean + std*3
    lower_limit = mean - std*3

    for x in range(0, len(hr_discrete)):
        if hr_discrete[x] > upper_limit:
            hr_discrete[x] = upper_limit
        elif hr_discrete[x] < lower_limit:
            hr_discrete[x] = lower_limit

    hrdata_res = np.array(signal.resample(hr_discrete, frames_vid))
    hrdata_res = np.array(mms.fit_transform(hrdata_res.reshape(-1,1))).flatten() # normalization
    
    working_data, measures = hp.process(hrdata_res, fps_vid, calc_freq=True)

    plot =  hp.plotter(working_data, measures, show=False, title = 'Heart Rate Signal and Peak Detection')
    path = path.replace('vid', 'plot_truthData').replace('.avi', '.jpg').replace('data', 'plot_truthData')
    plot.savefig(path)

    return working_data, measures

def dataSet_preprocess(vid, name):
    if name== "COHFACE":
        if os.path.exists(str(vid).replace(".avi", "_vid.hdf5")):
            os.remove(str(vid).replace(".avi", "_vid.hdf5"))
            print("deleted")
        if os.path.exists(str(vid).replace(".avi", "_dataFile.hdf5")):
            os.remove(str(vid).replace(".avi", "_dataFile.hdf5"))
            print("deleted")
            
        dXsub, fps = preprocess_raw_video(vid, 36)
        print(dXsub.shape)
        nframesPerVideo = dXsub.shape[0]

        # ground truth data:
        truth_data_path = str(vid).replace(".avi", ".hdf5")
        hf = h5py.File(truth_data_path, 'r')
        pulse = np.array(hf['pulse'])

        hf.close()  # close the hdf5 file

        return nframesPerVideo, fps, dXsub, pulse
    
    elif name=="UBFC_PHYS":
        if os.path.exists(str(vid).replace(".avi", "_data.hdf5")):
            os.remove(str(vid).replace(".avi", "_data.hdf5"))
            print("deleted")
        if os.path.exists(str(vid).replace(".avi", "_dataFile.hdf5")):
            os.remove(str(vid).replace(".avi", "_dataFile.hdf5"))
            print("deleted")
        if os.path.exists(str(vid).replace(".avi", "_dataFile.hdf5").replace('vid_', '')):
            os.rename(str(vid).replace(".avi", "_dataFile.hdf5").replace('vid_', ''), 
            str(vid).replace(".avi", "_dataFileAll.hdf5").replace('vid_', ''))
            print("Rename")

        dXsub, fps = preprocess_raw_video(vid, 36)
        print(dXsub.shape)
        nframesPerVideo = dXsub.shape[0]

        # ground truth data:
        truth_data_path = str(vid).replace(".avi", ".csv").replace('vid', 'bvp')
        hf = pd.read_csv(truth_data_path)
        pulse = np.array(hf)

        return nframesPerVideo, fps, dXsub, pulse


def build_h5py(vid, name):
    print("Dataset:   ", name)
    print("Current:   ", vid)
    nframesPerVideo, fps, dXsub, pulse = dataSet_preprocess(vid, name)
   
    # HR and HRV analysis
    working_data, measures = hr_analysis(vid, pulse, nframesPerVideo, fps)
    
    # Data for H5PY
    pulse_res = working_data['hr'] # resampled  and normalized HR
    nn_list = working_data['RR_list_cor'] # nn-intervals
    parameter = str(measures) # HR and HRV Parameter

    peak_list = working_data['peaklist']
    if not isinstance(peak_list, list):
        peak_list = peak_list.tolist()
    removed = working_data['removed_beats']
    for item in removed:
        peak_list.remove(item) # list with position of the peaks
   
    # new summary file
    # for UBFC_PHYS divide it in two 1min files
    if name == "UBFC_PHYS":
        newPath_name1 = str(vid).replace(".avi", "_0_dataFile.hdf5")
        newPath_name2 = str(vid).replace(".avi", "_1_dataFile.hdf5")
    else:   
        newPath_name = str(vid).replace(".avi", "_dataFile.hdf5")
    if name=="UBFC_PHYS":
        newPath_name = newPath_name.replace('vid_', '')
    data_file = h5py.File(newPath_name, 'a')
    
    data_file.create_dataset('data', data=dXsub)  # write the data to hdf5 file
    data_file.create_dataset('pulse', data=pulse_res)
    data_file.create_dataset('peaklist', data=peak_list)
    data_file.create_dataset('nn', data=nn_list)
    data_file.create_dataset('parameter', data=parameter)
    data_file.close()
    print("next")

    
def prepare_database(name, tasks, data_dir):
    if name == "UBFC_PHYS":
        taskList = list(range(1, tasks+1))
    elif name == "COHFACE":
        taskList = list(range(0, tasks))
    else: 
        print("Not implemented yet")
    subTrain, subTest = split_subj_(data_dir, name)
    print("subTrain:   ", subTrain)
    print("subTest:   ", subTest)
    video_path_list_tr  = sort_video_list_(data_dir, taskList, subTrain, name, True)
    video_path_list_test  = sort_video_list_(data_dir, taskList, subTest, name, False)
    video_path_list_tr =  list(itertools.chain(*video_path_list_tr))   
    video_path_list_test = list(itertools.chain(*video_path_list_test))

    for vid in video_path_list_tr:
        build_h5py(vid, name)
    for vid in video_path_list_test:
        build_h5py(vid, name)
    


data_dir = "E:/Databases"
#prepare_database("COHFACE", 4, data_dir)
prepare_database("UBFC_PHYS", 3, data_dir)

