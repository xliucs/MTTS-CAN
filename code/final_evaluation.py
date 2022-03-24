from aifc import Error
import numpy as np
import scipy.io
import xlsxwriter
from model import CAN, CAN_3D, PPTS_CAN, PTS_CAN, TS_CAN, Hybrid_CAN
import h5py
import os
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import scipy.stats as sc
from glob import glob

import heartpy as hp


def write_header(worksheet):
    header = ['Database', 'Subj/Task', 'HR-pred', 'HR-truth',
                'p','meanNN-pred', 'meanNN-truth', 'sdnn-pred', 'sdnn-truth', 
                'rmssd-pred', 'rmssd-truth', 'pNN50-pred', 'pNN50-truth',
                'LF-pred', 'LF-truth', 'HF-pred', 'HF-truth',
                'TP-pred', 'TP-truth', 'LF/HF-pred', 'LF/HF-truth',
                'sd1-pred', 'sd1_truth', 'sd2_pred', 'sd2_truth']
    for index in range(len(header)):
        worksheet.write(0,index, header[index])

def prepare_3D_CAN(dXsub):
    frame_depth = 10
    num_window = int(dXsub.shape[0]) - frame_depth + 1
    tempX = np.array([dXsub[f:f + frame_depth, :, :, :] # (491, 10, 36, 36 ,6) (169, 10, 36, 36, 6)
                    for f in range(num_window)])
    tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
    tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
    return tempX

def prepare_Hybrid_CAN(dXsub):
    frame_depth = 10
    num_window = int(dXsub.shape[0]) - frame_depth + 1
    tempX = np.array([dXsub[f:f + frame_depth, :, :, :] # (169, 10, 36, 36, 6)
                        for f in range(num_window)])
    tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
    tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
    motion_data = tempX[:, :, :, :, :3]
    apperance_data = np.average(tempX[:, :, :, :, -3:], axis=-2)
    return motion_data, apperance_data

def predict_vitals(worksheet, test_name, model_name, video_path, path_results):
    mms = MinMaxScaler()
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    batch_size = 100

    #### initialize Model ######
    try:
        model_checkpoint = os.path.join(path_results, test_name, "cv_0_epoch24_model.hdf5")
    except:
        model_checkpoint = os.path.join(path_results, test_name, "cv_0_epoch23_model.hdf5")
    batch_size = batch_size
    
    if model_name == "TS_CAN":
        model = TS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    elif model_name == "3D_CAN":
        model = CAN_3D(frame_depth, 32, 64, (img_rows, img_cols, frame_depth, 3))
    elif model_name == "CAN":
        model = CAN(32, 64, (img_rows, img_cols, 3))
    elif model_name == "Hybrid_CAN":
        model = Hybrid_CAN(frame_depth, 32, 64, (img_rows, img_cols, frame_depth, 3),
                            (img_rows, img_cols, 3))
    elif model_name == "PTS_CAN":
        model = PTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    elif model_name == "PPTS_CAN":
        model = PPTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3), parameter=['bpm', 'sdnn'])
    else: 
        raise NotImplementedError

    model.load_weights(model_checkpoint)
    ###### load video Data #######
    counter_video = 1
    old_database = "COH"
    for sample_data_path in video_path:
        print("path:  ",sample_data_path)
        dXsub, fs = preprocess_raw_video(sample_data_path, dim=36)
        print('dXsub shape', dXsub.shape, "fs: ", fs)
        
        if model_name == "PPTS_CAN":
            dXsub_len = (dXsub.shape[0] // (frame_depth*10))  * (frame_depth*10)
            dXsub = dXsub[:dXsub_len, :, :, :]
            
        else: 
            dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
            dXsub = dXsub[:dXsub_len, :, :, :]
        
        if model_name == "3D_CAN":
            dXsub = prepare_3D_CAN(dXsub)
            dXsub_len = (dXsub.shape[0] // (frame_depth))  * (frame_depth)
            dXsub = dXsub[:dXsub_len, :, :, :,:]
            yptest = model.predict((dXsub[:, :, :,: , :3], dXsub[:, :, :, : , -3:]), verbose=1)
        elif model_name == "Hybrid_CAN":
            dXsub1, dXsub2 = prepare_Hybrid_CAN(dXsub)
            dXsub_len1 = (dXsub1.shape[0] // (frame_depth*10))  * (frame_depth*10)
            dXsub1 = dXsub1[:dXsub_len1, :, :, :, :]
            dXsub_len2 = (dXsub2.shape[0] // (frame_depth*10))  * (frame_depth*10)
            dXsub2 = dXsub2[:dXsub_len2, :, :, :]
            yptest = model.predict((dXsub1, dXsub2), verbose=1)
        else:
            yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), verbose=1)
        if model_name != "PTS_CAN" and model_name != "PPTS_CAN":
            pulse_pred = yptest
        else:
            pulse_pred = yptest[0]
        
        pulse_pred = detrend(np.cumsum(pulse_pred), 100)
        [b_pulse_pred, a_pulse_pred] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_pred = scipy.signal.filtfilt(b_pulse_pred, a_pulse_pred, np.double(pulse_pred))
        pulse_pred = np.array(mms.fit_transform(pulse_pred.reshape(-1,1))).flatten()
        
        ##### ground truth data resampled  #######
        if(str(sample_data_path).find("COHFACE") > 0):
            database_name = "COH"
            truth_path = sample_data_path.replace(".avi", "_dataFile.hdf5")   # akutell für COHACE...
        elif(str(sample_data_path).find("UBFC-PHYS") > 0):
            database_name = "UB-Ph"
            truth_path = sample_data_path.replace("vid_", "").replace(".avi","_dataFile.hdf5")
        elif(str(sample_data_path).find("UBFC") > 0):
            database_name = "UBFC"
            truth_path = sample_data_path.replace("vid.avi", "dataFile.hdf5")
        else:
            return print("Error in finding the ground truth signal...")

        gound_truth_file = h5py.File(truth_path, "r")
        pulse_truth = gound_truth_file["pulse"]   ### range ground truth from 0 to 1
        pulse_truth = pulse_truth[0:dXsub_len]
        pulse_truth = detrend(np.cumsum(pulse_truth), 100)
        [b_pulse_tr, a_pulse_tr] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_truth = scipy.signal.filtfilt(b_pulse_tr, a_pulse_tr, np.double(pulse_truth))
        pulse_truth = np.array(mms.fit_transform(pulse_truth.reshape(-1,1))).flatten()
        ### same size #######
        if len(pulse_pred) > len(pulse_truth):
            pulse_pred = pulse_pred[:len(pulse_truth)]
        elif len(pulse_pred) < len(pulse_truth):
            pulse_truth = pulse_truth[:len(pulse_pred)]
        ########### Peaks ###########
        working_data_pred, measures_pred = hp.process(pulse_pred, fs, calc_freq=True)
        working_data_truth, measures_truth = hp.process(pulse_truth, fs, calc_freq=True)
        peaks_pred = working_data_pred['peaklist']
        peaks_truth = working_data_truth['peaklist']

        ######## name files #############
        if(str(sample_data_path).find("COHFACE") > 0):
            nmr = str(sample_data_path).find("COHFACE")
            nameStr = str(sample_data_path)[nmr + 7:].replace("\\", "-").replace("-data.avi", "")
        elif(str(sample_data_path).find("UBFC-PHYS") > 0):
            nmr = str(sample_data_path).find("UBFC-PHYS")
            nameStr = str(sample_data_path)[nmr + 12:].replace("\\", "-").replace("vid_", "").replace(".avi", "")
        elif(str(sample_data_path).find("UBFC") > 0):
            nmr = str(sample_data_path).find("UBFC")
            nameStr = str(sample_data_path)[nmr + 17:].replace("\\", "-").replace("vid.avi", "")
        else:
            raise ValueError
        ########## Plot ##################
        plt.figure() #subplot(211)
        plt.plot(pulse_pred, label='predicted PPG')
        plt.plot(peaks_truth, pulse_truth[peaks_truth], "x")
        plt.plot(peaks_pred, pulse_pred[peaks_pred], "o")
        plt.title('PPG prediction with ground truth signal')
        plt.ylabel("normalized Signal [a.u.]")
        plt.xlabel("time (samples)")
        plt.plot(pulse_truth, label='ground truth')
        plt.legend()
        name1 = database_name+nameStr + "_both.svg"
        plt.savefig(name1, format='svg')

        plt.figure()
        plt.subplot(211)
        plt.plot(pulse_truth, label='Ground truth')
        plt.plot(peaks_truth, pulse_truth[peaks_truth], "x")
        plt.ylabel("normalized Signal [a.u.]")
        plt.title('Ground truth')
        plt.subplot(212)
        plt.plot(pulse_pred, label='Prediction')
        plt.plot(peaks_pred, pulse_pred[peaks_pred], "x")
        plt.title("Predicted rPPG")
        plt.ylabel("normalized Signal [a.u.]")
        plt.xlabel("time (samples)")
        plt.legend()
        name2 =  database_name + nameStr + ".svg"
        plt.savefig(name2, format='svg')

        ########### IBI #############
        #ibi_truth = working_data_truth['RR_list_cor']
        #print(ibi_truth)
        #ibi_pred = working_data_pred['RR_list_cor']
        #print(ibi_pred)
        ######### HRV featurs ##############
        #print("HRV Truth:  ",measures_truth)
        #print("HRV Pred:  ", measures_pred)
        ######### Metrics ##############
        # MSE:
        MAE = metrics.mean_absolute_error(pulse_truth, pulse_pred)
        MSE = metrics.mean_squared_error(pulse_truth, pulse_pred)
        # RMSE:
        RMSE = metrics.mean_squared_error(pulse_truth, pulse_pred, squared=False)
        # Pearson correlation:
        p = sc.pearsonr(pulse_truth, pulse_pred)

        ####### Logging #############
        if database_name != old_database:
            counter_video += 1
        worksheet.write(counter_video,0, database_name)
        worksheet.write(counter_video,1, nameStr)
        worksheet.write(counter_video,2, measures_pred['bpm'])
        worksheet.write(counter_video,3, measures_truth['bpm'])
        worksheet.write(counter_video,4, p[0])
        worksheet.write(counter_video,5, measures_pred['ibi'])
        worksheet.write(counter_video,6, measures_truth['ibi'])
        worksheet.write(counter_video,7, measures_pred['sdnn'])
        worksheet.write(counter_video,8, measures_truth['sdnn'])
        worksheet.write(counter_video,9, measures_pred['rmssd'])
        worksheet.write(counter_video,10, measures_truth['rmssd'])
        worksheet.write(counter_video,11, measures_pred['pnn50'])
        worksheet.write(counter_video,12, measures_truth['pnn50'])
        worksheet.write(counter_video,13, measures_pred['lf'])
        worksheet.write(counter_video,14, measures_truth['lf'])
        worksheet.write(counter_video,15, measures_pred['hf'])
        worksheet.write(counter_video,16, measures_truth['hf'])
        worksheet.write(counter_video,17, measures_pred['p_total'])
        worksheet.write(counter_video,18, measures_truth['p_total'])
        worksheet.write(counter_video,19, measures_pred['lf/hf'])
        worksheet.write(counter_video,20, measures_truth['lf/hf'])
        worksheet.write(counter_video,21, measures_pred['sd1'])
        worksheet.write(counter_video,22, measures_truth['sd1'])
        worksheet.write(counter_video,23, measures_pred['sd2'])
        worksheet.write(counter_video,24, measures_truth['sd2'])

        counter_video += 1
        old_database = database_name
   
if __name__ == "__main__":
    path_results = "D:/Databases/4)Results/Version4/TS_Databases" #finalVersions"
    data_dir = "D:/Databases/3)Testing/"
    dir_names = glob(path_results + "*")
    test_names = []
    for dir in dir_names:
        split = dir.split("\\")
        test_names.append(split[len(split)-1])
    batch_size = 8
    
    video_path = glob(os.path.join(data_dir, "**/*", '*.avi'), recursive=True)
    
    save_dir = "D:/Databases/5)Evaluation/finalEvaluation"
    
    print("Models: ", test_names)
    for test_name in test_names:
        print("Current Modelname: ", test_name)
        if str(test_name).find("3D_CAN") >=0:
            model_name = "3D_CAN"
        elif str(test_name).find("Hybrid_CAN") >= 0:
            model_name = "Hybrid_CAN"
        elif str(test_name).find("TS_CAN") >= 0:
            model_name = "TS_CAN"
        elif str(test_name).find("PPTS") >= 0:
            model_name = "PPTS_CAN"
        elif str(test_name).find("PTS") >= 0:
            model_name = "PTS_CAN"
        else:
            if str(test_name).find("CAN") >= 0:
                model_name = "CAN"
            else: 
                raise Error("Model not found...")
                
        # neuer Ordner für Tests
        os.chdir(save_dir)
        try:
            os.makedirs(str(test_name))
        except:
            print("Directory exists...")
        save_path = os.path.join(save_dir, str(test_name))
        os.chdir(save_path)
        workbook = xlsxwriter.Workbook(test_name + ".xlsx")
        worksheet = workbook.add_worksheet("Results")
        write_header(worksheet)
        predict_vitals(worksheet, test_name, model_name, video_path, path_results)
        print("Ready with this model")
        workbook.close()




#python code/predict_vitals_new.py --video_path "D:\Databases\1)Training\COHFACE\1\1\data.avi" --trained_model ./cv_0_epoch24_model.hdf5
#./rPPG-checkpoints/testCohFace1/cv_0_epoch24_model.hdf5
#./rPPG-checkpoints/test1/cv_0_epoch04_model.hdf5'
#python code/predict_vitals_new.py --video_path "D:\Databases\1)Training\UBFC-PHYS\s1\vid_s1_T1.avi" --trained_model ./cv_0_epoch24_model.hdf5