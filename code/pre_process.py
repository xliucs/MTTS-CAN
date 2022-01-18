import glob
import os

import h5py
import numpy as np
import scipy.io
import pandas as pd


def get_nframe_video(path):
    temp_f1 = h5py.File(path, 'r')
    temp_dysub = np.array(temp_f1["dysub"])
    nframe_per_video = temp_dysub.shape[0]
    return nframe_per_video

def get_nframe_video_(path):
    temp_f1 = h5py.File(path, 'r')
    temp_data = np.array(temp_f1["data"])
    nframe_per_video = temp_data.shape[0]
    return nframe_per_video


def get_nframe_video_val(path):
    temp_f1 = scipy.io.loadmat(path)
    temp_dXsub = np.array(temp_f1["dXsub"])
    nframe_per_video = temp_dXsub.shape[0]
    return nframe_per_video


def split_subj(data_dir, cv_split, subNum): # trennen der Daten innerhalb 1 Subjekts...
    print(subNum)
    f3 = h5py.File( data_dir +'/s1/bvp_s1_T1.csv', 'r')# "/testSub.mat"
   # f4 = pd.read_csv(data_dir + '/s1/bvp_s1_T1.csv')
    M = np.transpose(np.array(f3["M"])).astype(np.bool) #? wieso als bool?
    subTrain = subNum[~M[:, cv_split]].tolist() # wieso nur cv_split?
    subTest = subNum[M[:, cv_split]].tolist()
    return subTrain, subTest


def take_last_ele(ele):
    ele = ele.split('.')[0][-2:]
    try:
        return int(ele[-2:])
    except ValueError:
        return int(ele[-1:])


def sort_video_list(data_dir, taskList, subTrain):
    final = []
    for p in subTrain:
        for t in taskList:
            x = glob.glob(os.path.join(data_dir, 'P' + str(p) + 'T' + str(t) + 'VideoB2*.mat'))
            x = sorted(x)
            x = sorted(x, key=take_last_ele)
            final.append(x)
    return final


def sort_video_list_(data_dir, taskList, subTrain, database_name, train):
    final = []
    if database_name == "UBFC_PHYS":
        if train:
            for p in subTrain:
                x = glob.glob(os.path.join(data_dir, 'Training/UBFC-PHYS/s' + str(p), 'vid_s*'))
                x = sorted(x)
                #x = sorted(x, key=take_last_ele)
                final.append(x)
        else:
           for p in subTrain:
                x = glob.glob(os.path.join(data_dir, 'Validation/UBFC-PHYS/s' + str(p), 'vid_s*'))
                x = sorted(x)
                #x = sorted(x, key=take_last_ele)
                final.append(x)

    elif database_name == "COHFACE":
        if train:
            for p in subTrain:
                for t in taskList:
                    x = glob.glob(os.path.join(data_dir, 'Training/COHFACE/', str(p), str(t), 'data.avi'))
                    x = sorted(x)
                    #x = sorted(x, key=take_last_ele)
                    final.append(x)
        else:
             for p in subTrain:
                for t in taskList:
                    x = glob.glob(os.path.join(data_dir, 'Validation/COHFACE/', str(p), str(t), 'data.avi'))
                    x = sorted(x)
                    #x = sorted(x, key=take_last_ele)
                    final.append(x)
    else: 
        print("not implemented yet.")
    return final

def sort_dataFile_list_(data_dir, taskList, subTrain, database_name, train):
    final = []
    if database_name == "UBFC_PHYS":
        if train:
            for p in subTrain:
                x = glob.glob(os.path.join(data_dir, '1)Training/UBFC-PHYS/s' + str(p), "s" + str(p) + "*"))
                x = sorted(x)
                #x = sorted(x, key=take_last_ele)
                final.append(x)
        else:
           for p in subTrain:
                x = glob.glob(os.path.join(data_dir, '2)Validation/UBFC-PHYS/s' + str(p), "s" + str(p) + "*"))
                x = sorted(x)
                #x = sorted(x, key=take_last_ele)
                final.append(x)

    elif database_name == "COHFACE":
        if train:
            for p in subTrain:
                for t in taskList:
                    x = glob.glob(os.path.join(data_dir, '1)Training/COHFACE/', str(p), str(t), 'data_datafile*'))
                    x = sorted(x)
                    #x = sorted(x, key=take_last_ele)
                    final.append(x)
        else:
             for p in subTrain:
                for t in taskList:
                    x = glob.glob(os.path.join(data_dir, '2)Validation/COHFACE/', str(p), str(t), 'data_datafile*'))
                    x = sorted(x)
                    #x = sorted(x, key=take_last_ele)
                    final.append(x)
    elif database_name == "MIX":
        for database in subTrain.keys():
            x = glob.glob(os.path.join(database, "**","*datafile.hdf5"),recursive=True)
            final.append(x)
        
    else: 
        print("not implemented yet.")
    return final


def split_subj_(data_dir, database): # trennen der Daten innerhalb 1 Subjekts...
    if database == "UBFC_PHYS":
        subTrain = np.array(range(1, 37)).tolist() #,37)).tolist()
        subTest = np.array(range(37,57)).tolist()
    elif database == "COHFACE":
        subTrain = np.array(range(1, 33)).tolist()# 33)).tolist()
        subTest = np.array(range(32,41)).tolist() # 41)).tolist()
    else:
        print("This Database isn't implemented yet.")
    return subTrain, subTest

def collect_subj(data_dir): # collecting all subject out of data_dir..
    
    path_tr = os.path.join(data_dir, "1)Training")
    path_val = os.path.join(data_dir, "2)Validation")
    databases_tr =glob.glob(os.path.join(path_tr, "*"))
    databases_val =glob.glob(os.path.join(path_val, "*"))
    subTrain = {}
    subTest = {}
    for database in databases_tr:
        subj_tr = os.listdir(database)
        subTrain[database] = subj_tr
    for database in databases_val:
        subj_val = os.listdir(database)
        subTest[database] = subj_val

    return subTrain, subTest
