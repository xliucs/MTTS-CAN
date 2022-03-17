'''
Training Script for Multi-Task Temporal Shift Attention Networks for On-Device
Contactless Vitals Measurement
Author: Xin Liu, Daniel McDuff
'''
# %%
from __future__ import print_function

import argparse
import itertools
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from xmlrpc.client import boolean
from losses import negPearsonLoss, negPearsonLoss_onlyPeaks, gaussian_loss, MAPE_parameter_loss, time_error_loss
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.python.keras.optimizers import adadelta_v2
from data_generator import DataGenerator
from model import HeartBeat, CAN, CAN_3D, Hybrid_CAN, TS_CAN, MTTS_CAN, \
    MT_Hybrid_CAN, MT_CAN_3D, MT_CAN, PTS_CAN, PPTS_CAN
from pre_process import split_subj_, sort_dataFile_list_, collect_subj

np.random.seed(100)  # for reproducibility
print("START!")
list_gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(list_gpu[0], enable=True)
tf.config.experimental.set_memory_growth(list_gpu[1], enable=True)
print(list_gpu)
tf.keras.backend.clear_session()
tf.autograph.set_verbosity(10)
print(tf.__version__)

# %%
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-exp', '--exp_name', type=str,
                    help='experiment name')
parser.add_argument('-i', '--data_dir', type=str, help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/home/quehl/Results/',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-a', '--nb_filters1', type=int, default=32,
                    help='number of convolutional filters to use')
parser.add_argument('-b', '--nb_filters2', type=int, default=64,
                    help='number of convolutional filters to use')
parser.add_argument('-c', '--dropout_rate1', type=float, default=0.25,
                    help='dropout rates')
parser.add_argument('-d', '--dropout_rate2', type=float, default=0.5,
                    help='dropout rates')
parser.add_argument('-l', '--lr', type=float, default=1.0,
                    help='learning rate')
parser.add_argument('-e', '--nb_dense', type=int, default=128,
                    help='number of dense units')
parser.add_argument('-f', '--cv_split', type=int, default=0,
                    help='cv_split')
parser.add_argument('-g', '--nb_epoch', type=int, default=24,
                    help='nb_epoch')
parser.add_argument('-t', '--nb_task', type=int, default=12,
                    help='nb_task')
parser.add_argument('-fd', '--frame_depth', type=int, default=10,
                    help='frame_depth for CAN_3D, TS_CAN, Hybrid_CAN')
parser.add_argument('-temp', '--temporal', type=str, default='PTS_CAN',
                    help='CAN, MT_CAN, CAN_3D, MT_CAN_3D, Hybrid_CAN, \
                    MT_Hybrid_CAN, TS_CAN, MTTS_CAN. PTS_CAN ')
parser.add_argument('-save', '--save_all', type=int, default=1,
                    help='save all or not')
parser.add_argument('-resp', '--respiration', type=int, default=0,
                    help='train with resp or not')
parser.add_argument('-database', '--database_name', type=str, 
                    default="MIX", help='Which database')  
parser.add_argument('-lf1', '--loss_function1', type=str, default="MSE") 
parser.add_argument('-lf2', '--loss_function2', type=str, default="MSE") 
parser.add_argument('-min', '--decrease_database', type=boolean, default=False)                       
parser.add_argument('-ml', '--maxFrames_video', type=int, default=2050, help="frames")
parser.add_argument('-p', '--parameter', default=None)

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

# %% Training


def train(args, subTrain, subTest, cv_split, img_rows=36, img_cols=36):
    print('================================')
    print('Train...')
    print('subTrain', subTrain)
    print('subTest', subTest)

    input_shape = (img_rows, img_cols, 3)
    maxLen_video = args.maxFrames_video

    path_of_video_tr = sort_dataFile_list_(args.data_dir, subTrain, args.database_name, trainMode=True)
    path_of_video_test = sort_dataFile_list_(args.data_dir, subTest, args.database_name, trainMode=False)

    #nframe_per_video = get_nframe_video_(path_of_video_tr[0])
    print('Train Length: ', len(path_of_video_tr))
    print('Test Length: ', len(path_of_video_test))
    if len(list_gpu) > 1:
        print("Using MultiWorkerMirroredStrategy")
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else: 
        print("Using MirroredStrategy")
        strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        
        if args.temporal == 'CAN' or args.temporal == 'MT_CAN':
            args.batch_size = 16
        elif args.temporal == 'CAN_3D' or args.temporal == 'MT_CAN_3D':
            args.batch_size = 2
        elif args.temporal == 'TS_CAN' or args.temporal == 'MTTS_CAN'\
            or  args.temporal == 'PTS_CAN':
            args.batch_size = 12#32
        elif args.temporal == 'PPTS_CAN':
            args.batch_size = 2
        elif args.temporal == 'Hybrid_CAN' or args.temporal == 'MT_Hybrid_CAN':
            args.batch_size = 2# 16
        else:
            raise ValueError('Unsupported Model Type!')

        if strategy.num_replicas_in_sync == 8:
            print('Using 8 GPUs for training!')
            args.batch_size = args.batch_size * 2
        elif strategy.num_replicas_in_sync == 2:
            print('Using 2 GPUs for training!')
            args.batch_size = args.batch_size // 2
        elif strategy.num_replicas_in_sync == 1:
            print('Using 1 GPU for training!')
            args.batch_size = 1#4
        elif strategy.num_replicas_in_sync == 4:
            print("Using 4 GPUs for training")
        else:
            raise Exception('Only supporting 4 GPUs or 8 GPUs now. Please adjust learning rate in the training script!')

        if args.temporal == 'CAN':
            print('Using CAN!')
            model = CAN(args.nb_filters1, args.nb_filters2, input_shape, dropout_rate1=args.dropout_rate1,
                        dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
        elif args.temporal == 'MT_CAN':
            print('Using MT_CAN!')
            model = MT_CAN(args.nb_filters1, args.nb_filters2, input_shape, dropout_rate1=args.dropout_rate1,
                           dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
        elif args.temporal == 'CAN_3D':
            print('Using CAN_3D!')
            input_shape = (img_rows, img_cols, args.frame_depth, 3)
            model = CAN_3D(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                           dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
        elif args.temporal == 'MT_CAN_3D':
            print('Using MT_CAN_3D!')
            input_shape = (img_rows, img_cols, args.frame_depth, 3)
            model = MT_CAN_3D(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                              dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2,
                              nb_dense=args.nb_dense)
        elif args.temporal == 'TS_CAN':
            print('Using TS_CAN!')
            input_shape = (img_rows, img_cols, 3)
            model = TS_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                           dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
        elif args.temporal == 'PTS_CAN':
            print('Using PTS_CAN: with PeakLocation!')
            input_shape = (img_rows, img_cols, 3)
            model = PTS_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                           dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
        elif args.temporal == 'PPTS_CAN':
            print('Using PPTS_CAN: with PeakLocation!')
            input_shape = (img_rows, img_cols, 3)
            args.parameter = str(args.parameter).split(",")
            model = PPTS_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                           dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense, parameter=args.parameter)
        elif args.temporal == 'MTTS_CAN':
            print('Using MTTS_CAN!')
            input_shape = (img_rows, img_cols, 3)
            model = MTTS_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                             dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
        elif args.temporal == 'Hybrid_CAN':
            print('Using Hybrid_CAN!')
            input_shape_motion = (img_rows, img_cols, args.frame_depth, 3)
            input_shape_app = (img_rows, img_cols, 3)
            model = Hybrid_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape_motion,
                               input_shape_app,
                               dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2,
                               nb_dense=args.nb_dense)
        elif args.temporal == 'MT_Hybrid_CAN':
            print('Using MT_Hybrid_CAN!')
            input_shape_motion = (img_rows, img_cols, args.frame_depth, 3)
            input_shape_app = (img_rows, img_cols, 3)
            model = MT_Hybrid_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape_motion,
                                  input_shape_app,
                                  dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2,
                                  nb_dense=args.nb_dense)
        else:
            raise ValueError('Unsupported Model Type!')

        optimizer = adadelta_v2.Adadelta(learning_rate=args.lr)
        if args.temporal == 'MTTS_CAN' or args.temporal == 'MT_Hybrid_CAN' or args.temporal == 'MT_CAN_3D' or \
                args.temporal == 'MT_CAN':
            losses = {"output_1": "mean_squared_error", "output_2": "mean_squared_error"}
            loss_weights = {"output_1": 1.0, "output_2": 1.0}
            model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)
        
        elif args.temporal == 'PTS_CAN':
            # output 1: rPPG Signal
            if args.loss_function1 == "MSE":
                loss1 = 'mean_squared_error'
                loss_weights1 = 1
            elif args.loss_function1 == "NegPea":
                loss1 = negPearsonLoss
                loss_weights1 = 1
            elif args.loss_function1 == "MSE_negPea":
                loss1a = 'mean_squared_error'
                loss1b = negPearsonLoss
                loss1 = [loss1a, loss1b]
                loss_weights1 = [1,1]
            elif args.loss_function1 == "Gauss_Peak":
                raise NotImplementedError
            # output 2: Gaussdistribution around peak locations or TimeError
            if args.loss_function2 == "MSE":
                loss2 = 'mean_squared_error'
                loss_weights2 = 1
            elif args.loss_function2 == "NegPea":
                loss2 = negPearsonLoss
                loss_weights2 = 1
                raise NotImplementedError
            elif args.loss_function2 == "Gauss_Peak":
                loss2 = gaussian_loss
                loss_weights2 = 1
            elif args.loss_function2 == "time_Error":
                loss2 = time_error_loss
                loss_weights2 = 1
                   
            losses = {"output_1": loss1, "output_2": loss2}
            loss_weights = {"output_1": loss_weights1, "output_2": loss_weights2}
            model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)
        
        elif args.temporal == 'PPTS_CAN':
            # output 1: rPPG Signal
            if args.loss_function1 == "MSE":
                loss1 = 'mean_squared_error'
                loss_weights1 = 1
            elif args.loss_function1 == "NegPea":
                loss1 = negPearsonLoss
                loss_weights1 = 1
            elif args.loss_function1 == "MSE_negPea":
                loss1a = 'mean_squared_error'
                loss1b = negPearsonLoss
                loss1 = [loss1a, loss1b]
                loss_weights1 = [1,1]
            elif args.loss_function1 == "Gauss_Peak":
                raise NotImplementedError
            # output 2: Gaussdistribution around peak locations or TimeError
            if args.loss_function2 == "MSE":
                loss2 = 'mean_squared_error'
                loss_weights2 = 1
            elif args.loss_function2 == "NegPea":
                loss2 = negPearsonLoss
                loss_weights2 = 1
                raise NotImplementedError
            elif args.loss_function2 == "Gauss_Peak":
                loss2 = gaussian_loss
                loss_weights2 = 1
            elif args.loss_function2 == "time_Error":
                loss2 = time_error_loss
                loss_weights2 = 1
            else: 
                raise NotImplementedError
            # output 3: different Parameter
            loss3 = MAPE_parameter_loss
            loss_weights3 = 1
                   
            losses = {"output_1": loss1, "output_2": loss2, "output_3": loss3}
            loss_weights = {"output_1": loss_weights1, "output_2": loss_weights2, "output_3": loss_weights3}
            model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)
        
        else:
            if args.loss_function1 == "MSE":
                model.compile(loss='mean_squared_error', optimizer=optimizer)
            elif args.loss_function1 == "negPea":
                print("negative Pearson Loss ")
                loss = negPearsonLoss
                model.compile(loss=loss, optimizer=optimizer)
            elif args.loss_function1 == "MSE_negPea":
                loss1 = 'mean_squared_error'
                loss2 = negPearsonLoss
                losses = [loss1, loss2]
                loss_weights = [1,1]
                model.compile(loss= losses, loss_weights=loss_weights, optimizer=optimizer)
            else:
                return ValueError('Unsupported Loss Function')

        print('learning rate: ', args.lr)
        print('batch size: ', args.batch_size)

        if args.loss_function2 == "time_Error":
            timeError = True
        else: 
            timeError = False

        # %% Create data genener
        training_generator = DataGenerator(path_of_video_tr, maxLen_video, (img_rows, img_cols),
                                           batch_size=args.batch_size, frame_depth=args.frame_depth,
                                           temporal=args.temporal, respiration=args.respiration, 
                                           database_name=args.database_name, time_error_loss=timeError,
                                           truth_parameter=args.parameter)
        validation_generator = DataGenerator(path_of_video_test, maxLen_video, (img_rows, img_cols),
                                             batch_size=args.batch_size, frame_depth=args.frame_depth,
                                             temporal=args.temporal, respiration=args.respiration,
                                            database_name=args.database_name, time_error_loss=timeError,
                                            truth_parameter=args.parameter)
      
        # %%  Checkpoint Folders
        checkpoint_folder = str(os.path.join(args.save_dir, args.exp_name))
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        cv_split_path = str(os.path.join(checkpoint_folder, "cv_" + str(cv_split)))

        # %% Callbacks
        if args.save_all == 1:
            save_best_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=cv_split_path + "_epoch{epoch:02d}_model.hdf5",
                save_best_only=False, verbose=1)
        else:
            save_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cv_split_path + "_last_model.hdf5",
                                                                    save_best_only=False, verbose=1)
        csv_logger = tf.keras.callbacks.CSVLogger(filename=cv_split_path + '_train_loss_log.csv')
        hb_callback = HeartBeat(training_generator, validation_generator, args, str(cv_split), checkpoint_folder)
        
        # %% Model Training and Saving Results
        history = model.fit(x=training_generator, validation_data=validation_generator, epochs=args.nb_epoch, 
                    verbose=1, shuffle=True, callbacks=[csv_logger, save_best_callback, hb_callback], validation_freq=4)

        val_loss_history = history.history['val_loss']
        val_loss = np.array(val_loss_history)
        np.savetxt((cv_split_path + '_val_loss_log.csv'), val_loss, delimiter=",")

        score = model.evaluate_generator(generator=validation_generator, verbose=1)

        print('****************************************')
        if args.temporal == 'MTTS_CAN' or args.temporal == 'MT_Hybrid_CAN' or args.temporal == 'MT_CAN_3D' \
                or args.temporal == 'MT_CAN':
            print('Average Test Score: ', score[0])
            print('PPG Test Score: ', score[1])
            print('Respiration Test Score: ', score[2])
        else:
            print('Test score:', score)
        print('****************************************')
        print('Start saving predicitions from the last epoch')

        training_generator = DataGenerator(path_of_video_tr, maxLen_video, (img_rows, img_cols),
                                           batch_size=args.batch_size, frame_depth=args.frame_depth,
                                           temporal=args.temporal, respiration=args.respiration, shuffle=False,
                                           database_name=args.database_name, time_error_loss=timeError,
                                           truth_parameter=args.parameter)

        validation_generator = DataGenerator(path_of_video_test, maxLen_video, (img_rows, img_cols),
                                             batch_size=args.batch_size, frame_depth=args.frame_depth,
                                             temporal=args.temporal, respiration=args.respiration, shuffle=False,
                                             database_name=args.database_name, time_error_loss=timeError,
                                            truth_parameter=args.parameter)

        yptrain = model.predict(training_generator, verbose=1)
        scipy.io.savemat(checkpoint_folder + '/yptrain_best_' + '_cv' + str(cv_split) + '.mat',
                         mdict={'yptrain': yptrain})
        yptest = model.predict(validation_generator, verbose=1)
        scipy.io.savemat(checkpoint_folder + '/yptest_best_' + '_cv' + str(cv_split) + '.mat',
                         mdict={'yptest': yptest})

        file = open(checkpoint_folder + "/log.txt","w")
        file.write("LogFile\n\n")
        file.write("Name:  "), file.write(args.exp_name)
        file.write("\nModel:   "), file.write(args.temporal)
        file.write("\nBatch Size:   "), file.write(str(args.batch_size))
        file.write("\nLoss Function (output1):  "), file.write(args.loss_function1)
        file.write("\nLoss Function (output2):  "), file.write(args.loss_function2)
        file.write("\nLoss Function (output3):  "), file.write("MAPE")
        file.write("\nMax Frames Video: "), file.write(str(args.maxFrames_video))
        file.write("\nLearningrate:   "), file.write(str(args.lr))
        file.write("\nTrain Subjects:  "), file.write(str(subTrain))
        file.write("\nValidation Subjects:  "), file.write(str(subTest))
        file.close()

        print('Finish saving the results from the last epoch')


# %% Training

print('Using Split ', str(args.cv_split))
print("DatabaseName:  ", args.database_name)
# Mix1: COHFACE and UBFC-Phys 
# Mix2: COHFACE and UBFC-rPPG
if args.database_name != "MIX1" and args.database_name != "MIX2":
    subTrain, subTest = split_subj_(args.data_dir, args.database_name)
else:
    subTrain, subTest = collect_subj(args.data_dir, args.database_name)

if args.decrease_database == True:
    if args.database_name == "COHFACE":
        subTrain = subTrain[0:10]
        subTest = subTest[0:3]
    elif args.database_name == "UBFC_PHYS":
        subTrain = subTrain[0:25]
        subTest = subTest[0:10]
    elif args.database_name == "MIX1":
        for key in subTrain.keys():
            subTrain[key] = subTrain[key][0:6]
        for key in subTest.keys():
            subTest[key] = subTest[key][0:3]

train(args, subTrain, subTest, args.cv_split)


