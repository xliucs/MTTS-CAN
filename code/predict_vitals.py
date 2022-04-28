from scipy import signal
import tensorflow as tf
import numpy as np
import scipy.io
import sys
import argparse
sys.path.append('../')
from model import TS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend
from hrvanalysis import get_time_domain_features, get_frequency_domain_features

def predict_vitals(args):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    #model_checkpoint = './mtts_can.hdf5'
    model_checkpoint = args.trained_model
    batch_size = args.batch_size
    sample_data_path = args.video_path
    print("path:  ",sample_data_path)

    dXsub, fs = preprocess_raw_video(sample_data_path, dim=36)
    print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = TS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred = yptest#[0]
    pulse_pred = (pulse_pred - pulse_pred.min())/(pulse_pred.max() - pulse_pred.min()) * 2 -1
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
    
    ##### ground truth data resampled  #######
    if(str(sample_data_path).find("COHFACE") >= 0):
        truth_path = args.video_path.replace(".avi", "_dataFile.hdf5")   # akutell für COHACE...
    elif(str(sample_data_path).find("UBFC-PHYS")>= 0):
        truth_path = args.video_path.replace("vid_", "").replace(".avi","_dataFile.hdf5")
    elif(str(sample_data_path).find("UBFC")>= 0):
        truth_path = args.video_path.replace("vid.avi","dataFile.hdf5")
    else:
        return print("Error in finding the ground truth signal...")
    gound_truth_file = h5py.File(truth_path, "r")
    pulse_truth = gound_truth_file["pulse"]
    pulse_truth = detrend(np.cumsum(pulse_truth), 100)
    [b_pulse_tr, a_pulse_tr] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_truth = scipy.signal.filtfilt(b_pulse_tr, a_pulse_tr, np.double(pulse_truth))
    ### range ground truth from -1 to 1
    pulse_truth = (pulse_truth - pulse_truth.min())/(pulse_truth.max() - pulse_truth.min()) * 2 -1
    
    #pulse_pred = pulse_pred[5:]
    ########### Peaks ###########
    peaks_truth, peaks_ = np.array(signal.find_peaks(pulse_truth, prominence=0.5))
    peaks_pred, b  = np.array(signal.find_peaks(pulse_pred, prominence=0.2))

     ########## Plot ##################
    plt.figure() #subplot(211)
    plt.plot(pulse_pred, label='Prediction')
    plt.plot(peaks_truth, pulse_truth[peaks_truth], "x")
    plt.plot(peaks_pred, pulse_pred[peaks_pred], "o")
    plt.title('Pulse Prediction')
    plt.plot(pulse_truth, label='ground truth')
    plt.legend()

    plt.figure()
    plt.subplot(211)
    plt.plot(pulse_truth, label='Ground truth')
    plt.plot(peaks_truth, pulse_truth[peaks_truth], "x")
    plt.ylabel("normalized Signal")
    plt.title('Ground truth')
    plt.subplot(212)
    plt.plot(pulse_pred, label='Prediction')
    plt.plot(peaks_pred, pulse_pred[peaks_pred], "x")
    plt.title("Prediction")
    plt.ylabel("normalized Signal")
    plt.legend()
    plt.show()

    ########### IBI #############
    ibi_truth = np.diff(peaks_truth)*(1000/fs)
    print(ibi_truth)
    ibi_pred = np.diff(peaks_pred)*(1000/fs)
    print(ibi_pred)
    ######### HRV featurs ##############
    time_domain_features = get_time_domain_features(ibi_truth)
    time_domain_features_pred = get_time_domain_features(ibi_pred)
    print(time_domain_features)
    print(time_domain_features_pred)
    freq_domain_features = get_frequency_domain_features(ibi_truth)
    freq_domain_features_pred = get_frequency_domain_features(ibi_pred)
    print(freq_domain_features)
    print(freq_domain_features_pred)
    ####### Logging #############
    # neuer Ordner für Tests
    file = open(str(sample_data_path).replace(".avi", "_result.txt"),"w")
    file.write("LogFile\n\n")
    file.write("IBI: "), file.write(str(ibi_pred))
    file.write("\nTime-domain features: "), file.write(str(time_domain_features_pred))
    file.write("\nFrequency-domain features: "), file.write(str(freq_domain_features_pred))

    file.write("\n\n\nGround truth infos!")
    file.write("\nTime-domain features: "), file.write(str(time_domain_features))
    file.write("\nFrequency-domain features: "), file.write(str(freq_domain_features))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    parser.add_argument('--trained_model', type=str, default = "D:/Databases/4)Results/Version4/TS_Databases/TS_CAN_COHFACE_2GPU/cv_0_epoch24_model.hdf5", help='path to trained model')

    args = parser.parse_args()

    predict_vitals(args)


#python code/predict_vitals.py --video_path "C:\Users\sarah\OneDrive\Desktop\UBFC\DATASET_2\subject4\vid.avi" --trained_model "D:\Databases\4)Results\Version4\TS_Databases\TS_CAN_COHFACE_2GPU\cv_0_epoch23_model.hdf5"
#./rPPG-checkpoints/testCohFace1/cv_0_epoch24_model.hdf5
#./rPPG-checkpoints/test1/cv_0_epoch04_model.hdf5'