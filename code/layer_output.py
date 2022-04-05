

import os
from model import CAN_3D, PPTS_CAN, PTS_CAN, TS_CAN
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf



def gaussian_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, (-1,))
    return -tf.reduce_sum(tf.abs(y_true*y_pred))

path_of_video_tr = ["D:/Databases/1)Training/COHFACE/5/1/data_dataFile.hdf5"]

model = PPTS_CAN(10, 32, 64, (36,36,3),
                           dropout_rate1=0.25, dropout_rate2=0.5, nb_dense=128, parameter=['bpm', 'sdnn', 'pnn50', 'lf_hf'])
training_generator = DataGenerator(path_of_video_tr, 2100, (36, 36),
                                           batch_size=1, frame_depth=10,
                                           temporal="PPTS_CAN", respiration=False, database_name="COHFACE", 
                                           time_error_loss=True, truth_parameter=['bpm', 'sdnn', 'pnn50', 'lf_hf'])

inp = model.input   # input placeholder
#model.summary()
model_checkpoint = os.path.join("D:/Databases/4)Results/altesPPTS/PPTS_CAN_bpm_sdnn/cv_0_epoch24_model.hdf5")
model.load_weights(model_checkpoint)
#outputs = [layer.output for layer in model.layers]          # all layer outputs
#functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

# Testing
test = training_generator.data_generation(path_of_video_tr) 
output = model(test)

#### test new output ######
pred = tf.reshape(output[2], (-1))
truth = tf.reshape(test[1][2], (-1))
AE = tf.abs(pred - truth)/truth
loss = tf.reduce_mean(AE)
diff = pred - truth
#print(output[1])

plt.plot(output[0], label="Prediction")
plt.plot(test[1][0], label="truth")
plt.legend()
plt.show()

# plt.figure()
# plt.subplot(211)
# plt.title('Predicted signal example')
# plt.plot(output[0][100:500], label='output 1')
# plt.ylabel("rPPG [a.u.]")
# plt.legend(loc="upper right")
# plt.subplot(212)
# plt.plot(output[1][100:500], label='output 2')
# #plt.ylabel("[a.u.]")
# plt.legend(loc="upper right")
# plt.show()

# plt.figure()
# plt.subplot(211)
# plt.title('Ground truth example')
# plt.plot(test[1][0][100:500], label='truth data 1')
# plt.ylabel("rPPG [a.u.]")
# plt.legend(loc="upper right")
# plt.subplot(212)
# plt.plot(test[1][1][100:500], label='truth data 2')
# plt.xlabel("time (samples)")
# plt.legend(loc="upper right")
# plt.show()

# y_true = test[1][1]
# y_pred = output[1]
# loss = gaussian_loss(y_true, y_pred)
# mult = y_true * tf.reshape(y_pred, (-1,))
# plt.plot(mult)
# plt.title("Multiplication of y_true and y_pred")
# plt.ylabel("[a.u.]")
# plt.xlabel("time (samples)")
# plt.show()

#layer_outs = [func([test, 1.]) for func in functors]
#print(layer_outs)


