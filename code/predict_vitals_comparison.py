from msilib.schema import Error
import numpy as np
import scipy.io
import sys
import argparse
sys.path.append('../')
from model import Attention_mask, MTTS_CAN, TS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend
from sklearn.preprocessing import MinMaxScaler

import heartpy as hp
import os

def predict_vitals(args):
    mms = MinMaxScaler()
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = args.trained_model
    batch_size = args.batch_size
    sample_data_path = args.video_path
    print("path:  ",sample_data_path)

    ts_can_COHFACE =  os.path.join(args.trained_model, 'TS_CAN_COHFACE_2GPU\\cv_0_epoch24_model.hdf5' )
    ts_can_UBFC_PHYS = os.path.join(args.trained_model,"TS_CAN_UBFC_PHYS/cv_0_epoch24_model.hdf5")
    ts_can_MIX = os.path.join(args.trained_model, "TS_CAN_MIX_2GPU/cv_0_epoch24_model.hdf5")
    
    dXsub, fs = preprocess_raw_video(sample_data_path, dim=36)
    print("PROCESSES", fs)
    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    model_COHFACE = TS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model_COHFACE.load_weights(ts_can_COHFACE)
    model_MIX = TS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model_MIX.load_weights(ts_can_MIX)
    model_UBFC_PHYS = TS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model_UBFC_PHYS.load_weights(ts_can_UBFC_PHYS)

    yptest_COHFACE = model_COHFACE.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)
    yptest_MIX = model_MIX.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)
    yptest_UBFC_PHYS = model_UBFC_PHYS.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred_COHFACE = detrend(np.cumsum(yptest_COHFACE), 100)
    [b_pulse_pred, a_pulse_pred] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred_COHFACE = scipy.signal.filtfilt(b_pulse_pred, a_pulse_pred, np.double(pulse_pred_COHFACE))
    pulse_pred_COHFACE = np.array(mms.fit_transform(pulse_pred_COHFACE.reshape(-1,1))).flatten()

    pulse_pred_MIX = detrend(np.cumsum(yptest_MIX), 100)
    [b_pulse_pred, a_pulse_pred] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred_MIX = scipy.signal.filtfilt(b_pulse_pred, a_pulse_pred, np.double(pulse_pred_MIX))
    pulse_pred_MIX = np.array(mms.fit_transform(pulse_pred_MIX.reshape(-1,1))).flatten()

    pulse_pred_UBFC_PHYS = detrend(np.cumsum(yptest_UBFC_PHYS), 100)
    [b_pulse_pred, a_pulse_pred] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred_UBFC_PHYS = scipy.signal.filtfilt(b_pulse_pred, a_pulse_pred, np.double(pulse_pred_UBFC_PHYS))
    pulse_pred_UBFC_PHYS = np.array(mms.fit_transform(pulse_pred_UBFC_PHYS.reshape(-1,1))).flatten()

    
    ##### ground truth data resampled  #######
    if(str(sample_data_path).find("COHFACE") >=0):
        truth_path = args.video_path.replace(".avi", "_dataFile.hdf5")
    elif(str(sample_data_path).find("UBFC-PHYS") >= 0):
        truth_path = args.video_path.replace("vid_", "").replace(".avi","_dataFile.hdf5")
    else:
        raise ValueError("Error in finding the ground truth signal...")
    gound_truth_file = h5py.File(truth_path, "r")
    pulse_truth = gound_truth_file["pulse"]   ### range ground truth from 0 to 1
    pulse_truth = detrend(np.cumsum(pulse_truth), 100)
    [b_pulse_tr, a_pulse_tr] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_truth = scipy.signal.filtfilt(b_pulse_tr, a_pulse_tr, np.double(pulse_truth))
    pulse_truth = np.array(mms.fit_transform(pulse_truth.reshape(-1,1))).flatten()

    ########### Peaks ###########
    working_data_pred_COH, measures_pred_COH = hp.process(pulse_pred_COHFACE, fs, calc_freq=True)
    working_data_pred_MIX, measures_pred_MIX = hp.process(pulse_pred_MIX, fs, calc_freq=True)
    working_data_pred_UBFC, measures_pred_UBFC = hp.process(pulse_pred_UBFC_PHYS, fs, calc_freq=True)
    working_data_truth, measures_truth = hp.process(pulse_truth, fs, calc_freq=True)
    
    peaks_pred_COH = working_data_pred_COH['peaklist']
    peaks_pred_MIX = working_data_pred_MIX['peaklist']
    peaks_pred_UBFC = working_data_pred_UBFC['peaklist']
    peaks_truth = working_data_truth['peaklist']

     ########## Plot ##################
    print("FIGURE")
    plt.figure() #subplot(211)
    plt.plot(pulse_pred_COHFACE, "--", color="k", linewidth=1, label='Prediction COHFACE')
    plt.plot(peaks_pred_COH, pulse_pred_COHFACE[peaks_pred_COH], "x", color="k")
    plt.plot(pulse_pred_MIX, ":",color="grey", linewidth=1.1,  label='Prediction MIX')
    plt.plot(peaks_pred_MIX, pulse_pred_MIX[peaks_pred_MIX], "x", color="grey")
    plt.plot(pulse_pred_UBFC_PHYS, "-.", color="silver", linewidth=1.1, label='Prediction UBFC-PHYS')
    plt.plot(peaks_pred_UBFC, pulse_pred_UBFC_PHYS[peaks_pred_UBFC], "x", color="silver")
    plt.ylabel("normalized Signal [a.u.]")
    plt.xlabel("time (samples)")

    plt.plot(peaks_truth, pulse_truth[peaks_truth], "x", color="b")
    plt.title('Example: Participant out of UBFC-PHYS database')
    plt.plot(pulse_truth, "#005AA9",  label='ground truth')
    plt.legend()
    plt.show()

    
    
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(pulse_truth, label='Ground truth')
    # plt.plot(peaks_truth, pulse_truth[peaks_truth], "x")
    # plt.ylabel("normalized Signal")
    # plt.title('Ground truth')
    # plt.subplot(212)
    # plt.plot(pulse_pred_COHFACE, label='Prediction')
    # plt.plot(peaks_pred_COH, pulse_pred_COHFACE[peaks_pred_COH], "x")
    # plt.title("Prediction")
    # plt.ylabel("normalized Signal")
    # plt.legend()
    # plt.show()

    ########### IBI #############
    ibi_truth = working_data_truth['RR_list_cor']
    print(ibi_truth)
    ibi_pred_COH = working_data_pred_COH['RR_list_cor']
    print(ibi_pred_COH)
    ibi_pred_MIX = working_data_pred_MIX['RR_list_cor']
    print(ibi_pred_MIX)
    ibi_pred_UBFC = working_data_pred_UBFC['RR_list_cor']
    print(ibi_pred_UBFC)
    ######### HRV featurs ##############
    print("HRV Truth:  ",measures_truth)
    print("HRV Pred COHFACE:  ", measures_pred_COH)
    print("HRV Pred MIX:  ", measures_pred_MIX)
    print("HRV Pred UBFC:  ", measures_pred_UBFC)
    ####### Logging #############
    # neuer Ordner für Tests
    # file = open(str(sample_data_path).replace(".avi", "comparisonALL_result.txt"),"w")
    # file.write("LogFile\n\n")
    # file.write("\nCOHFACE:")
    # file.write("\nIBI: "), file.write(str(ibi_pred_COH))
    # file.write("\nHR and HRVfeatures: "), file.write(str(measures_pred_COH))

    # file.write("\nMIX:")
    # file.write("\nIBI: "), file.write(str(ibi_pred_MIX))
    # file.write("\nHR and HRVfeatures: "), file.write(str(measures_pred_MIX))

    # file.write("\nUBFC-PHYS:")
    # file.write("\nIBI: "), file.write(str(ibi_pred_UBFC))
    # file.write("\nHR and HRVfeatures: "), file.write(str(measures_pred_UBFC))

    # file.write("\n\n\nGround truth infos!")
    # file.write("\nHR and HRV features: "), file.write(str(measures_truth))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    parser.add_argument('--trained_model', type=str, default = './rPPG-checkpoints/testCohFace1/cv_0_epoch24_model.hdf5', help='path to trained model')

    args = parser.parse_args()

    predict_vitals(args)


#python code/predict_vitals_comparison.py --video_path "D:/Databases/1)Training/COHFACE/5/1/data.avi" --trained_model "D:\Databases\4)Results\"
#python code/predict_vitals_comparison.py --video_path "D:/Databases/2)Validation/UBFC-PHYS/s44/vid_s44_T1.avi" --trained_model "D:\Databases\4)Results\actual_models"