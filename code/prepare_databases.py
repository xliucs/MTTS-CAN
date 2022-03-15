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

def hr_analysis(path, hr_discrete, frames_vid, fps_vid, nr=None):
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
    if nr ==None:
        path = path.replace('vid', 'plot_truthData').replace('.avi', '.jpg').replace('data', 'plot_truthData')
    else:
        path = path.replace('vid', 'plot_truthData_' + str(nr)).replace('.avi', '.jpg').replace('data', 'plot_truthData')
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
        print(dXsub.shape, "  fps: ", fps)
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
        if os.path.exists(str(vid).replace(".avi", "_dataFile.hdf5").replace('vid_', '')):
            os.rename(str(vid).replace(".avi", "_dataFile.hdf5").replace('vid_', ''), str(vid).replace(".avi", "_dataFile.hdf5").replace('vid_', 'ALL_'))
            print("renamed")
        if os.path.exists(str(vid).replace(".avi", "_dataFileAll.hdf5").replace('vid_', '')):
            os.remove(str(vid).replace(".avi", "_dataFileAll.hdf5").replace('vid_', ''))
            print("deleted")

        dXsub, fps = preprocess_raw_video(vid, 36)
        print(dXsub.shape, "   fps: ", fps)
        nframesPerVideo = dXsub.shape[0]

        # ground truth data:
        truth_data_path = str(vid).replace(".avi", ".csv").replace('vid', 'bvp')
        hf = pd.read_csv(truth_data_path)
        pulse = np.array(hf)

        return nframesPerVideo, fps, dXsub, pulse
    if name == "UBFC":
        if os.path.exists(str(vid).replace(".avi", "_vid.hdf5")):
            os.remove(str(vid).replace(".avi", "_vid.hdf5"))
            print("deleted")
        if os.path.exists(str(vid).replace(".avi", "_dataFile.hdf5")):
            os.remove(str(vid).replace(".avi", "_dataFile.hdf5"))
            print("deleted")
            
        dXsub, fps = preprocess_raw_video(vid, 36)
        print(dXsub.shape, "  fps: ", fps)
        nframesPerVideo = dXsub.shape[0]

        # ground truth data:
        truth_data_path = str(vid).replace(".avi", ".txt").replace('vid', 'ground_truth')
        data = open(truth_data_path, 'r').read()
        data = str(data).split("  ")
        data = data[1:]
        pulse = np.array(list(map(float, data[0: nframesPerVideo])))

        return nframesPerVideo, fps, dXsub, pulse

def process_save(nframesPerVideo, fps, dXsub, pulse, vid):
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
    
    ##### save data ######
    newPath_name = str(vid).replace(".avi", "_dataFile.hdf5")
    if (str(vid).find("UBFC") >=0):
        newPath_name = newPath_name.replace("vid_", "")
    data_file = h5py.File(newPath_name, 'a')
    data_file.create_dataset('data', data=dXsub)  # write the data to hdf5 file
    data_file.create_dataset('pulse', data=pulse_res)
    data_file.create_dataset('peaklist', data=peak_list)
    data_file.create_dataset('nn', data=nn_list)
    data_file.create_dataset('parameter', data=parameter)
    data_file.close()

def process_save_UBFC(nframesPerVideo, fps, dXsub, pulse, vid):
    ##### split in 3 one minute parts ###########
    frames_per_dataPacket = int(fps*60)
    frames_per_pulsePacket = int(64*60)
    dXsub_1 = dXsub[0:frames_per_dataPacket,:,:,:]
    dXsub_2 = dXsub[frames_per_dataPacket: frames_per_dataPacket*2,:,:,:]
    dXsub_3 = dXsub[frames_per_dataPacket*2:,:,:,:]
    pulse_1 = pulse[0:frames_per_pulsePacket,:]
    pulse_2 = pulse[frames_per_pulsePacket:frames_per_pulsePacket*2,:]
    pulse_3 = pulse[frames_per_pulsePacket*2:,:]
    # HR and HRV analysis
    working_data1, measures1 = hr_analysis(vid, pulse_1, dXsub_1.shape[0], fps, nr=1)
    working_data2, measures2 = hr_analysis(vid, pulse_2, dXsub_2.shape[0], fps, nr=2)
    working_data3, measures3 = hr_analysis(vid, pulse_3, dXsub_3.shape[0], fps, nr=3)
    
    # Data for H5PY
    pulse_res1 = working_data1['hr'] # resampled  and normalized HR
    pulse_res2 = working_data2['hr'] # resampled  and normalized HR
    pulse_res3 = working_data3['hr'] # resampled  and normalized HR
    nn_list1 = working_data1['RR_list_cor'] # nn-intervals
    nn_list2 = working_data2['RR_list_cor'] # nn-intervals
    nn_list3 = working_data3['RR_list_cor'] # nn-intervals
    parameter1 = str(measures1) # HR and HRV Parameter
    parameter2 = str(measures2) # HR and HRV Parameter
    parameter3 = str(measures3) # HR and HRV Parameter

    peak_list1 = working_data1['peaklist']
    if not isinstance(peak_list1, list):
        peak_list1 = peak_list1.tolist()
    removed1 = working_data1['removed_beats']
    for item in removed1:
        peak_list1.remove(item) # list with position of the peaks
    
    peak_list2 = working_data2['peaklist']
    if not isinstance(peak_list2, list):
        peak_list2 = peak_list2.tolist()
    removed2 = working_data2['removed_beats']
    for item in removed2:
        peak_list2.remove(item) # list with position of the peaks
    
    peak_list3 = working_data3['peaklist']
    if not isinstance(peak_list3, list):
        peak_list3 = peak_list3.tolist()
    removed3 = working_data3['removed_beats']
    for item in removed3:
        peak_list3.remove(item) # list with position of the peaks


    newPath_name1 = str(vid).replace(".avi", "_0_dataFile.hdf5").replace('vid_', '')
    newPath_name2 = str(vid).replace(".avi", "_1_dataFile.hdf5").replace('vid_', '')
    newPath_name3 = str(vid).replace(".avi", "_2_dataFile.hdf5").replace('vid_', '')
    data_file1 = h5py.File(newPath_name1, 'a')
    data_file1.create_dataset('data', data=dXsub_1)  # write the data to hdf5 file
    data_file1.create_dataset('pulse', data=pulse_res1)
    data_file1.create_dataset('peaklist', data=peak_list1)
    data_file1.create_dataset('nn', data=nn_list1)
    data_file1.create_dataset('parameter', data=parameter1)
    data_file1.close()
    data_file2 = h5py.File(newPath_name2, 'a')
    data_file2.create_dataset('data', data=dXsub_2)  # write the data to hdf5 file
    data_file2.create_dataset('pulse', data=pulse_res2)
    data_file2.create_dataset('peaklist', data=peak_list2)
    data_file2.create_dataset('nn', data=nn_list2)
    data_file2.create_dataset('parameter', data=parameter2)
    data_file2.close()
    data_file3 = h5py.File(newPath_name3, 'a')
    data_file3.create_dataset('data', data=dXsub_3)  # write the data to hdf5 file
    data_file3.create_dataset('pulse', data=pulse_res3)
    data_file3.create_dataset('peaklist', data=peak_list3)
    data_file3.create_dataset('nn', data=nn_list3)
    data_file3.create_dataset('parameter', data=parameter3)
    data_file3.close()

def build_h5py(vid, name):
    print("Dataset:   ", name)
    print("Current:   ", vid)
    nframesPerVideo, fps, dXsub, pulse = dataSet_preprocess(vid, name)
    if name != "UBFC_PHYS":
        process_save(nframesPerVideo, fps, dXsub, pulse, vid)
    else:
        process_save_UBFC(nframesPerVideo, fps, dXsub, pulse, vid)   
        
    print("next")

    
def prepare_database(name, tasks, data_dir):
    if name == "UBFC_PHYS":
        taskList = list(range(1, tasks+1))
    elif name == "COHFACE":
        taskList = list(range(0, tasks))
    elif name == "UBFC":
        taskList = [0]
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
    


#data_dir = "D:/Databases"
#prepare_database("COHFACE", 4, data_dir)
data_dir = "/mnt/share/StudiShare/sarah/Databases/"

data_dir = "C:/Users/sarah/OneDrive/Desktop/UBFC/DATASET_2"
prepare_database("UBFC", 1, data_dir)

