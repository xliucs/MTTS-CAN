import glob
import itertools
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
                
                x = glob.glob(os.path.join(data_dir, '1)Training/UBFC-PHYS/s' + str(p), 'vid_s*'))
                x = sorted(x)
                #x = sorted(x, key=take_last_ele)
                final.append(x)
        else:
           for p in subTrain:
                x = glob.glob(os.path.join(data_dir, '2)Validation/UBFC-PHYS/s' + str(p), 'vid_s*'))
                x = sorted(x)
                #x = sorted(x, key=take_last_ele)
                final.append(x)

    elif database_name == "COHFACE":
        if train:
            for p in subTrain:
                for t in taskList:
                    x = glob.glob(os.path.join(data_dir, '1)Training/COHFACE/', str(p), str(t), 'data.avi'))
                    x = sorted(x)
                    #x = sorted(x, key=take_last_ele)
                    final.append(x)
        else:
             for p in subTrain:
                for t in taskList:
                    x = glob.glob(os.path.join(data_dir, '2)Validation/COHFACE/', str(p), str(t), 'data.avi'))
                    x = sorted(x)
                    #x = sorted(x, key=take_last_ele)
                    final.append(x)
    elif database_name == "UBFC":
        if train:
            x = glob.glob(os.path.join(data_dir, "**/", 'vid.avi'), recursive=True)
            x = sorted(x)
            #x = sorted(x, key=take_last_ele)
            final.append(x)
    else: 
        print("not implemented yet.")
    return final

def sort_dataFile_list_(data_dir, subTrain, database_name, trainMode):
    if database_name == "UBFC_PHYS":
        final = dataFiles_UBFC_PHYS(data_dir, subTrain, trainMode, mode=1)
        final = list(itertools.chain(*final))
    elif database_name == "COHFACE":
        taskList = [0, 1, 2, 3]
        final = dataFile_COHFACE(data_dir, taskList, subTrain, trainMode)
        final = list(itertools.chain(*final))
    elif database_name == "UBFC":
        final = dataFile_UBFC(data_dir, trainMode)
        final = list(itertools.chain(*final))
    elif database_name == "MIX1":
        for database in subTrain.keys():
            final = []
            if(str(database).find("UBFC") >= 0):
                finalPart1  = dataFiles_UBFC_PHYS(data_dir, subTrain[database], trainMode, mode=0)
                finalPart1 = list(itertools.chain(*finalPart1))
            elif str(database).find("COHFACE") >= 0:
                taskList = [0, 1, 2, 3]
                finalPart2 = dataFile_COHFACE(data_dir, taskList, subTrain[database], trainMode)
                finalPart2 = list(itertools.chain(*finalPart2))
            else:
                raise NotImplementedError
        final = finalPart1 + finalPart2
    elif database_name == "MIX2":
        for database in subTrain.keys():
            final = []
            if(str(database).find("UBFC") >= 0):
                finalPart1  = dataFile_UBFC(data_dir, trainMode)
                finalPart1 = list(itertools.chain(*finalPart1))
            elif str(database).find("COHFACE") >= 0:
                taskList = [0, 1, 2, 3]
                finalPart2 = dataFile_COHFACE(data_dir, taskList, subTrain[database], trainMode)
                finalPart2 = list(itertools.chain(*finalPart2))
            else:
                raise NotImplementedError
        final = finalPart1 + finalPart2
    else: 
        print("not implemented yet.")
    return final

def dataFile_COHFACE(data_dir, taskList, subTrain, train):
    final = []
    if train:
        for p in subTrain:
            for t in taskList:
                x = glob.glob(os.path.join(data_dir, '1)Training/COHFACE', str(p), str(t), '*dataFile.hdf5'))
                x = sorted(x)
                    #x = sorted(x, key=take_last_ele)
                final.append(x)
    else:
         for p in subTrain:
            for t in taskList:
                x = glob.glob(os.path.join(data_dir, '2)Validation/COHFACE/', str(p), str(t), '*dataFile.hdf5'))
                x = sorted(x)
                    #x = sorted(x, key=take_last_ele)
                final.append(x)
    return final

def dataFiles_UBFC_PHYS(data_dir, subTrain, train, mode):
    final = []
    if mode == 1:
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
    else:
        if train:
            for p in subTrain:
                x = glob.glob(os.path.join(data_dir, '1)Training/UBFC-PHYS/' + str(p), str(p) + "*"))
                x = sorted(x)
                #x = sorted(x, key=take_last_ele)
                final.append(x)
        else:
            for p in subTrain:
                x = glob.glob(os.path.join(data_dir, '2)Validation/UBFC-PHYS/' + str(p), str(p) + "*"))
                x = sorted(x)
                #x = sorted(x, key=take_last_ele)
                final.append(x)
    return final

def dataFile_UBFC(data_dir, train):
    final = []
    if train:
        x = glob.glob(os.path.join(data_dir,'1)Training/UBFC', "**/", 'dataFile.hdf5'), recursive=True)
        x = sorted(x)
        #x = sorted(x, key=take_last_ele)
        final.append(x)
    else:
        x = glob.glob(os.path.join(data_dir,'2)Validation/UBFC', "**/", 'dataFile.hdf5'), recursive=True)
        x = sorted(x)
        #x = sorted(x, key=take_last_ele)
        final.append(x)
    return final


def split_subj_(data_dir, database): # trennen der Daten innerhalb 1 Subjekts...
    if database == "UBFC_PHYS":
        subTrain = np.array(range(1, 37)).tolist() #,37)).tolist()
        subTest = np.array(range(37,57)).tolist()
    elif database == "COHFACE":
        subTrain = np.array(range(1, 33)).tolist()# 33)).tolist()
        subTest = np.array(range(32,41)).tolist() # 41)).tolist()
    elif database == "UBFC":
        subTrain = np.array(range(1,34))
        subTest = np.array([range(34,42)])
    else:
        print("This Database isn't implemented yet.")
    return subTrain, subTest

def collect_subj(data_dir, database_name): # collecting all subject out of data_dir..
    
    path_tr = os.path.join(data_dir, "1)Training")
    path_val = os.path.join(data_dir, "2)Validation")
    mix1 = ['COHFACE', 'UBFC-PHYS']
    mix2 = ['COHFACE', 'UBFC']
    if database_name == "MIX1":
        mix = mix1
    elif database_name == "MIX2":
        mix = mix2
    subTrain = {}
    subTest = {}
    for database in mix:
        subj_tr = os.listdir(os.path.join(path_tr, database))
        subTrain[database] = subj_tr
        subj_val = os.listdir(os.path.join(path_val, database))
        subTest[database] = subj_val

    return subTrain, subTest
