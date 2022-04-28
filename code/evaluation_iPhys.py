import numpy as np
import scipy.io
import xlsxwriter
import h5py
import os
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_frames, preprocess_raw_video, detrend
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import scipy.stats as sc
from glob import glob
from scipy import signal

import heartpy as hp


def write_header(worksheet):
    header = ['Database', 'Subj/Task', 'HR-pred', 'HR-truth',
                'p','meanNN-pred', 'meanNN-truth', 'sdnn-pred', 'sdnn-truth', 
                'rmssd-pred', 'rmssd-truth', 'pNN50-pred', 'pNN50-truth',
                'LF-pred', 'LF-truth', 'HF-pred', 'HF-truth',
                'TP-pred', 'TP-truth', 'LF/HF-pred', 'LF/HF-truth',
                'sd1-pred', 'sd1_truth', 'sd2_pred', 'sd2_truth', 'MAE']
    for index in range(len(header)):
        worksheet.write(0,index, header[index])
        
def predict_vitals(worksheet, video_path, save_dir):
    mms = MinMaxScaler()
    
    ###### load video Data #######
    counter_video = 1
    old_database = "COH"
    for sample_data_path in video_path:
        
        print("path:  ",sample_data_path)
        pulse_pred = open(sample_data_path, 'r').read()
        pulse_pred = str(pulse_pred).split("\n")
        pulse_pred = pulse_pred[:-1]
        pulse_pred = np.array(list(map(float, pulse_pred)))

        mean = pulse_pred.mean()
        std = np.std(pulse_pred)
        upper_limit = mean + std*3
        lower_limit = mean - std*3
        for x in range(0, len(pulse_pred)):
            if pulse_pred[x] > upper_limit:
                pulse_pred[x] = upper_limit
            elif pulse_pred[x] < lower_limit:
                pulse_pred[x] = lower_limit
        pulse_pred = np.array(mms.fit_transform(pulse_pred.reshape(-1,1))).flatten() # normalization
        
        
        ##### ground truth data resampled  #######
        data_path = "default"
        if(str(sample_data_path).find("GC") > 0):
            method = "GC"
            data_path = sample_data_path.replace("_GC", "")
        elif(str(sample_data_path).find("ICA_POH") > 0):
            method = "ICA"
            data_path = sample_data_path.replace("_ICA_POH", "")
        elif(str(sample_data_path).find("CHROM") > 0):
            method = "CHROM"
            data_path = sample_data_path.replace("_CHROM", "")
        else:
            raise print("ERROR")

        if(str(sample_data_path).find("COHFACE") > 0):
            database_name = "COH"
            fs = 20
            truth_path = data_path.replace(".txt", "_dataFile.hdf5")   # akutell für COHACE...
        elif(str(sample_data_path).find("UBFC-PHYS") > 0):
            database_name = "UB-Ph"
            fs = 35
            truth_path = data_path.replace("vid_", "").replace(".txt","_dataFile.hdf5")
        elif(str(sample_data_path).find("UBFC") > 0):
            database_name = "UBFC"
            fs = 30
            truth_path = data_path.replace("vid.txt", "dataFile.hdf5")
        elif(str(sample_data_path).find("BP4D") > 0):
            fs = 25
            database_name = "BP4D"
            truth_path = data_path + "/BP_mmHg.txt"
        else:
            return print("Error in finding the ground truth signal...")
       
        if database_name != "BP4D":
            gound_truth_file = h5py.File(truth_path, "r")
            pulse_truth = gound_truth_file["pulse"]   ### range ground truth from 0 to 1
            pulse_truth = pulse_truth[0:len(pulse_pred)]
        else:
            data = open(truth_path, 'r').read()
            data = str(data).split("\n")
            pulse_truth = np.array(list(map(float, data[0:-1])))
            mean = pulse_truth.mean()
            std = np.std(pulse_truth)
            upper_limit = mean + std*3
            lower_limit = mean - std*3
            for x in range(0, len(pulse_truth)):
                if pulse_truth[x] > upper_limit:
                    pulse_truth[x] = upper_limit
                elif pulse_truth[x] < lower_limit:
                    pulse_truth[x] = lower_limit

            pulse_truth = np.array(signal.resample(pulse_truth, len(pulse_pred)))
            pulse_truth = np.array(mms.fit_transform(pulse_truth.reshape(-1,1))).flatten() # normalization
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
        try:
            working_data_pred, measures_pred = hp.process(pulse_pred, fs, calc_freq=True)
            working_data_truth, measures_truth = hp.process(pulse_truth, fs, calc_freq=True)
        except:
            continue
        peaks_pred = working_data_pred['peaklist']
        peaks_truth = working_data_truth['peaklist']

        ######## name files #############
        if(str(data_path).find("COHFACE") > 0):
            nmr = str(data_path).find("COHFACE")
            nameStr = str(data_path)[nmr + 7:].replace("\\", "-").replace("-data.txt", "")
        elif(str(data_path).find("UBFC-PHYS") > 0):
            nmr = str(data_path).find("UBFC-PHYS")
            nameStr = str(data_path)[nmr + 12:].replace("\\", "-").replace("vid_", "").replace(".txt", "")
        elif(str(data_path).find("UBFC") > 0):
            nmr = str(data_path).find("UBFC")
            nameStr = str(data_path)[nmr + 5:].replace("\\", "-").replace("vid.txt", "")
        elif(str(data_path).find("BP4D") > 0):
            nmr = str(data_path).find("BP4D")
            nameStr = str(data_path)[nmr + 5:].replace("\\", "-")
        else:
            raise ValueError
        ########## Plot ##################
        peaks_pred_new = []
        for peak in peaks_pred:
            if (peak > 400 and peak <700):
                peaks_pred_new.append(peak-400)
        peaks_truth_new = []
        for peak in peaks_truth:
            if (peak > 400 and peak <700):
                peaks_truth_new.append(peak-400)
        plt.figure() #subplot(211)
        plt.plot(pulse_pred[400:700], "#E6001A", label='rPPG signal')
        plt.plot(peaks_truth_new, pulse_truth[400:700][peaks_truth_new], "x", color="#005AA9")
        plt.plot(peaks_pred_new, pulse_pred[400:700][peaks_pred_new], "x", color ='#E6001A')
        plt.title('rPPG signal with ground truth')
        plt.ylabel("normalized Signal [a.u.]")
        plt.xlabel("time (samples)")
        plt.plot(pulse_truth[400:700], '#005AA9', linewidth=0.9, label='ground truth')
        plt.legend()
        plt.savefig(save_dir + database_name+ nameStr + method + "_both.svg", format="svg")

        plt.figure()
        plt.subplot(211)
        plt.plot(pulse_truth[400:700],"#004E8A", label='Ground truth')
        plt.plot(peaks_truth_new, pulse_truth[400:700][peaks_truth_new], "x", color="#004E8A")
        plt.ylabel("normalized Signal [a.u.]")
        plt.title('Ground truth')
        plt.subplot(212)
        plt.plot(pulse_pred[400:700], "#004E8A",label='Prediction')
        plt.plot(peaks_pred_new, pulse_pred[400:700][peaks_pred_new],"x", color="#004E8A")
        plt.title("Predicted rPPG")
        plt.ylabel("normalized Signal [a.u.]")
        plt.xlabel("time (samples)")
        plt.legend()
        plt.savefig(save_dir + database_name+ nameStr + method +".svg", format="svg")

        ######### Metrics ##############
        # MSE:
        MAE = metrics.mean_absolute_error(pulse_truth, pulse_pred)
        # RMSE:
        RMSE = metrics.mean_squared_error(pulse_truth, pulse_pred, squared=False)
        # Pearson correlation:
        p = sc.pearsonr(pulse_truth, pulse_pred)

        ####### Logging #############
        if database_name != old_database:
            counter_video += 1
        worksheet.write(counter_video,0, database_name)
        worksheet.write(counter_video,1, nameStr)
        worksheet.write(counter_video,2,  str(measures_pred['bpm']))
        worksheet.write(counter_video,3,  str(measures_truth['bpm']))
        worksheet.write(counter_video,4, p[0])
        worksheet.write(counter_video,5,  str(measures_pred['ibi']))
        worksheet.write(counter_video,6,  str(measures_truth['ibi']))
        worksheet.write(counter_video,7,  str(measures_pred['sdnn']))
        worksheet.write(counter_video,8,  str(measures_truth['sdnn']))
        worksheet.write(counter_video,9,  str(measures_pred['rmssd']))
        worksheet.write(counter_video,10,  str(measures_truth['rmssd']))
        worksheet.write(counter_video,11,  str(measures_pred['pnn50']))
        worksheet.write(counter_video,12,  str(measures_truth['pnn50']))
        worksheet.write(counter_video,13, str(measures_pred['lf']))
        worksheet.write(counter_video,14, str(measures_truth['lf']))
        worksheet.write(counter_video,15, str(measures_pred['hf']))
        worksheet.write(counter_video,16, str(measures_truth['hf']))
        try:
            worksheet.write(counter_video,17,str(measures_pred['p_total']))
            worksheet.write(counter_video,18,str(measures_truth['p_total']))
        except:
            pass
        try:
            worksheet.write(counter_video,19, str(measures_pred['lf/hf']))
            worksheet.write(counter_video,20, str(measures_truth['lf/hf']))
            worksheet.write(counter_video,21, str(measures_pred['sd1']))
            worksheet.write(counter_video,22, str(measures_truth['sd1']))
            worksheet.write(counter_video,23, str(measures_pred['sd2']))
            worksheet.write(counter_video,24, str(measures_truth['sd2']))
            worksheet.write(counter_video,25, MAE)
        except:
            pass

        counter_video += 1
        old_database = database_name
   
if __name__ == "__main__":
    data_dir = 'D:/Databases/3)Testing/'
    
    GC_path = glob(os.path.join(data_dir, "**/*", '*GC.txt'), recursive=True)
    ICA_path = glob(os.path.join(data_dir, "**/*", '*ICA_POH.txt'), recursive=True)
    CHROM_path = glob(os.path.join(data_dir, "**/*", '*CHROM.txt'), recursive=True)
    
    
    save_dir = 'D:/Databases/5)Evaluation/Test/'
   
    workbook = xlsxwriter.Workbook(save_dir + "Result_iPhys" + ".xlsx")
    worksheet_GC = workbook.add_worksheet("GC")
    write_header(worksheet_GC)
    predict_vitals(worksheet_GC, GC_path, save_dir)
    print("Ready with this model")
    worksheet_ICA = workbook.add_worksheet("ICA")
    write_header(worksheet_ICA)
    predict_vitals(worksheet_ICA, ICA_path, save_dir)
    print("Ready with this model")
    worksheet_CHROM = workbook.add_worksheet("CHROM")
    write_header(worksheet_CHROM)
    predict_vitals(worksheet_CHROM, CHROM_path, save_dir)
    print("Ready with this model")
    workbook.close()
