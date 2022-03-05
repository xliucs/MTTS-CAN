

from sklearn.preprocessing import MinMaxScaler
from model import PTS_CAN, TS_CAN
from tensorflow import keras
from keras import backend as K
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def gaussian_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, (-1,))
    return -tf.reduce_sum(tf.abs(y_true*y_pred))

path_of_video_tr = ["D:/Databases/1)Training/COHFACE/1/0/data_dataFile.hdf5"]

model = PTS_CAN(10, 32, 64, (36,36,3),
                           dropout_rate1=0.25, dropout_rate2=0.5, nb_dense=128)
training_generator = DataGenerator(path_of_video_tr, 2100, (36, 36),
                                           batch_size=4, frame_depth=10,
                                           temporal="PTS_CAN", respiration=False, database_name="MIX", time_error_loss=True)

inp = model.input   # input placeholder
model.summary()
#outputs = [layer.output for layer in model.layers]          # all layer outputs
#functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

# Testing
test = training_generator.data_generation(path_of_video_tr) 
output = model(test, training=True)
print(output[1])
plt.figure()
plt.subplot(211)
plt.title('Predicted signal example')
plt.plot(output[0][100:500], label='output 0')
plt.ylabel("rPPG [a.u.]")
plt.legend(loc="upper right")
plt.subplot(212)
plt.plot(output[1][100:500], label='output 1')
#plt.ylabel("[a.u.]")
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.subplot(211)
plt.title('Ground truth example')
plt.plot(test[1][0][100:500], label='label 0')
plt.ylabel("rPPG [a.u.]")
plt.legend(loc="upper right")
plt.subplot(212)
plt.plot(test[1][1][100:500], label='label 1')
plt.xlabel("time (samples)")
plt.legend(loc="upper right")
plt.show()

y_true = test[1][1]
y_pred = output[1]
loss = gaussian_loss(y_true, y_pred)
mult = y_true * tf.reshape(y_pred, (-1,))
plt.plot(mult)
plt.title("Multiplication of y_true and y_pred")
plt.ylabel("[a.u.]")
plt.xlabel("time (samples)")
plt.show()

print(loss)
print("c")
#layer_outs = [func([test, 1.]) for func in functors]
#print(layer_outs)