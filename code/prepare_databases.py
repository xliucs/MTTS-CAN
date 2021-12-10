from cv2 import dct
from scipy.sparse import data
from pre_process import sort_video_list_, split_subj_
from inference_preprocess import preprocess_raw_video

import h5py
import itertools
import numpy as np
from scipy import signal
import os

def build_h5py_COHFACE(vid):
    if os.path.exists(str(vid).replace(".avi", "_vid.hdf5")):
        os.remove(str(vid).replace(".avi", "_vid.hdf5"))
        print("deleted")
        
    dXsub = preprocess_raw_video(vid, 36)
    print(dXsub.shape)
    nframesPerVideo = dXsub.shape[0]

    # ground truth data:
    truth_data_path = str(vid).replace(".avi", ".hdf5")
    hf = h5py.File(truth_data_path, 'r')
    pulse = hf["pulse"]
    respiration = hf["respiration"]
    
    pulse_resampled = signal.resample(pulse,nframesPerVideo)
    respiration_resampled = signal.resample(respiration, nframesPerVideo)

    # new summary file
    newPath_name = str(vid).replace(".avi", "_vid.hdf5")
    data_file = h5py.File(newPath_name, 'a')
    
    data_file.create_dataset('data', data=dXsub)  # write the data to hdf5 file
    data_file.create_dataset('pulse', data=pulse_resampled)
    data_file.create_dataset('respiration', data=respiration_resampled)
    hf.close()  # close the hdf5 file
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
    video_path_list_tr  = sort_video_list_(data_dir, taskList, subTrain, name, True)
    video_path_list_test  = sort_video_list_(data_dir, taskList, subTest, name, False)
    video_path_list_tr =  list(itertools.chain(*video_path_list_tr))
    video_path_list_test = list(itertools.chain(*video_path_list_test))

    if name == "COHFACE":
        for vid in video_path_list_tr:
            build_h5py_COHFACE(vid)
        for vid in video_path_list_test:
            build_h5py_COHFACE(vid)



# data_dir = "E:/Databases"
# tasks = 3
# prepare_database("UBFC_PHYS", tasks, data_dir)

data_dir = "E:/Databases"
tasks = 4
prepare_database("COHFACE", tasks, data_dir)
