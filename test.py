import math
from cv2 import threshold
import numpy as np
import tensorflow as tf
import h5py

import heartpy as hp
import matplotlib.pyplot as plt
def get_peaks(y):
        # y: (N,1)
        data_reshaped = tf.reshape(y, (1, -1, 1)) # (1, N, 1)
        max_pooled_in_tensor =  tf.nn.max_pool(data_reshaped, (20,), 1,'SAME')
        maxima = tf.equal(data_reshaped,max_pooled_in_tensor) # (1, N, 1)
        maxima = tf.cast(maxima, tf.float32)
        maxima = tf.squeeze(maxima) # (N,1)
        maxima = tf.reshape(maxima, (-1,1))
        #peaks = tf.where(maxima) # now only the Peak Indices (A, 3)
        #
        # peaks = tf.reshape(peaks, (-1,)) # (A,1)

        return maxima
print(get_peaks(1))

def gauss(x, sigma, mu):
    return math.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))


a = [2,28,132,163,174,196, 48,70,91,112]
b = [50]#[3,25, 50, 77, 91]

temp = np.zeros(100)
temp2 = np.zeros(100)

for k in b:
    temp2[k] = 1
plt.plot(temp2)
plt.show()

for i in b:
    mu = i
    sigma = 3
    for j in range(i-sigma*3, i+sigma*3):
        temp[j] = gauss(j, sigma, mu)

pred = np.zeros(100)
pred[45] = 1

plt.plot(temp)
plt.plot(pred)
plt.show()

def gaussian_loss(y_true, y_pred):
    return -tf.reduce_sum(y_true*y_pred)

loss = gaussian_loss(temp,pred)


print(loss)