'''
Script to predict a video over a defined model. 
Subsequent saving of the model outputs and 
analyses of the heartpy module.
input example: 
python code/predict_vitals_oneVideo.py --video_path "path-to-video" 
        --trained_model "D:\Databases\4)Results\Version5\TS_CAN\cv_0_epoch24_model.hdf5"
        --model_name "TS_CAN"
PPTS_CAN example:
        --parameter "bpm,sdnn"
'''

from aifc import Error
import argparse
import os
import numpy as np
import scipy.io
from model import CAN, CAN_3D, PPTS_CAN, PTS_CAN, TS_CAN, Hybrid_CAN
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_frames, preprocess_raw_video, detrend
from sklearn.preprocessing import MinMaxScaler
import heartpy as hp

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

def predict_vitals(args):
    model_checkpoint = args.trained_model
    sample_data_path = args.video_path
    model_name = args.model_name
    save_dir = args.save_dir
    parameter_PPTS = str(args.parameter).split(",")
    mms = MinMaxScaler()
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    # neuer Ordner für Tests
    os.chdir(save_dir)
    try:
        os.makedirs(str(model_name))
    except:
        print("Directory exists...")
    save_dir = os.path.join(save_dir, str(model_name))

    print("path:  ",sample_data_path)
    if sample_data_path[-4:] == ".avi":
        dXsub, fps = preprocess_raw_video(sample_data_path, dim=36)
    elif sample_data_path[-4:] == ".mp4":
        dXsub, fps = preprocess_raw_video(sample_data_path, dim=36)
    else: 
        dXsub, fps = preprocess_raw_frames(sample_data_path, dim=36)
    print('dXsub shape', dXsub.shape, "fps: ", fps)
    
    if model_name == "PPTS_CAN":
        dXsub_len = (dXsub.shape[0] // (frame_depth*10))  * (frame_depth*10)
        dXsub = dXsub[:dXsub_len, :, :, :]
    else: 
        dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
        dXsub = dXsub[:dXsub_len, :, :, :]
    
    if model_name == "TS_CAN":
        model = TS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    elif model_name == "3D_CAN":
        model = CAN_3D(frame_depth, 32, 64, (img_rows, img_cols, frame_depth, 3))
        dXsub = prepare_3D_CAN(dXsub)
        dXsub_len = (dXsub.shape[0] // (frame_depth))  * (frame_depth)
        dXsub = dXsub[:dXsub_len, :, :, :,:]
    elif model_name == "CAN":
        model = CAN(32, 64, (img_rows, img_cols, 3))
    elif model_name == "Hybrid_CAN":
        model = Hybrid_CAN(frame_depth, 32, 64, (img_rows, img_cols, frame_depth, 3),
                            (img_rows, img_cols, 3))
        dXsub1, dXsub2 = prepare_Hybrid_CAN(dXsub)
        dXsub_len1 = (dXsub1.shape[0] // (frame_depth))  * (frame_depth)
        dXsub1 = dXsub1[:dXsub_len1, :, :, :,:]
        dXsub_len2 = (dXsub2.shape[0] // (frame_depth))  * (frame_depth)
        dXsub2 = dXsub2[:dXsub_len2, :, :, :]
    elif model_name == "PTS_CAN":
        model = PTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    elif model_name == "PPTS_CAN":
        model = PPTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3), parameter=parameter_PPTS)
    else: 
        raise NotImplementedError


    model.load_weights(model_checkpoint)
    if model_name == "3D_CAN":
        yptest = model.predict((dXsub[:, :, :,: , :3], dXsub[:, :, :, : , -3:]))
    elif model_name == "Hybrid_CAN":
        yptest = model.predict((dXsub1, dXsub2))
    else:
        yptest = model((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), training=False)
    if model_name == "3D_CAN" or model_name == "Hybrid_CAN":
        pulse_pred = yptest[:,0]
    elif model_name != "PTS_CAN" and model_name != "PPTS_CAN":
        pulse_pred = yptest
        
    else:
        pulse_pred = yptest[0]

    if model_name == "PPTS_CAN":
        parameter_out = yptest[2]
     
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse_pred, a_pulse_pred] = butter(1, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse_pred, a_pulse_pred, np.double(pulse_pred))
    pulse_pred = np.array(mms.fit_transform(pulse_pred.reshape(-1,1))).flatten()
    
    ########### Peaks ###########
    working_data_pred, measures_pred = hp.process(pulse_pred, fps, calc_freq=True)
    peaks_pred = working_data_pred['peaklist']

    ######## x-axis: time #########
    duration_vid = dXsub_len/fps
    x_axis = np.linspace(0, duration_vid, dXsub_len)

    ########## Plot ##################
    peaks_pred_new = []
    for peak in peaks_pred:
        if (peak > 400 and peak <700):
            peaks_pred_new.append(peak-400)
    
    path_plot = save_dir + "/plot.png"
    print(path_plot)
   
    plt.figure() #subplot(211)
    plt.plot(x_axis, pulse_pred, "#E6001A", label='rPPG signal')
    plt.plot(x_axis[peaks_pred], pulse_pred[peaks_pred], "x", color ='#E6001A')
    plt.title('rPPG signal')
    plt.ylabel("normalized Signal [a.u.]")
    plt.xlabel("time (s)")
    plt.legend()
    plt.savefig(path_plot)

    file_rPPG = open(str(save_dir) + "/rPPG_out.txt","w")
    for value in pulse_pred:
        file_rPPG.write(str(value))#
        file_rPPG.write("\n")
    file_rPPG.close()
    if model_name =="PPTS_CAN":
        file_parameter = open(str(save_dir) + "/parameter_out.txt","w")
        file_parameter.write(str(parameter_out))
        file_parameter.close()
    file_hrAnalysis = open(str(save_dir) + "/HRVAnalysis.txt","w")
    file_hrAnalysis.write(str(measures_pred))
    file_hrAnalysis.close()


   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--save_dir', type=str, help='save dir path')
    parser.add_argument('--trained_model', type=str, default = "D:/Databases/4)Results/Version4/TS_Databases/TS_CAN_COHFACE_2GPU/cv_0_epoch24_model.hdf5", help='path to trained model')
    parser.add_argument('--model_name', type=str, help='name of model (TS_CAN, PTS_CAN,...)')
    parser.add_argument('--parameter', type=str, help='parameter for PPTS_CAN ("bpm,sdnn")')

    args = parser.parse_args()

    predict_vitals(args)

#python code/predict_vitals_oneVideo.py --video_path "C:\Users\sarah\OneDrive\Desktop\UBFC\DATASET_2\subject4\vid.avi" --trained_model "D:\Databases\4)Results\Version5\TS_CAN\cv_0_epoch24_model.hdf5" --model_name "TS_CAN" --save_dir "D:\Databases\Test"