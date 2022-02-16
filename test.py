import numpy as np
import tensorflow as tf
import h5py

import heartpy as hp

a = [2,28, 48,70,91,112,132,163,174,196]
b = [3, 6,28, 50, 91, 112, 150, 180]


x = tf.convert_to_tensor(a, dtype=tf.float64)
y = tf.convert_to_tensor(b, dtype=tf.float64)

x = tf.reshape(x, (1,-1,1))
x= tf.reshape(x, (-1))

data_reshaped = tf.reshape(y, (1, -1, 1)) # (1, N, 1)
max_pooled_in_tensor = tf.nn.pool(data_reshaped, window_shape=(20,), pooling_type='MAX', padding='SAME')
maxima = tf.equal(data_reshaped,max_pooled_in_tensor) # (1, N, 1)
maxima = tf.cast(maxima, tf.float64)
maxima = tf.squeeze(maxima) # (N,1)
peaks = tf.where(maxima)
v = tf.range(0, tf.size(y))
maxima = tf.cast(maxima, tf.bool)
peaks2 = tf.boolean_mask(v, maxima)



def true_fn():
    return min
def false_fn():
    return -1
max_offset = 10
mask = tf.cast(tf.zeros(tf.size(x)),tf.bool) # tensor with size of x (truth data)
# check which peaks of truth are recognized in pred
for item in y: # items of predicion
    diff = tf.abs(x - item) # diff of truth data and item
    min = tf.reduce_min(diff) # minimum of diff
    min = tf.cond(tf.less(min, max_offset), true_fn, false_fn)
    temp_mask = tf.equal(min, diff)
    mask = tf.logical_or(mask, temp_mask)

x = tf.boolean_mask(x, mask)
# check if outliners are in pred
mask = tf.cast(tf.zeros(tf.size(y)), tf.bool)
for item in x: 
    diff = tf.abs(y - item) # diff of truth data and item
    min = tf.reduce_min(diff) # minimum of diff
    min = tf.cond(tf.less(min, max_offset), true_fn, false_fn)
    temp_mask = tf.equal(min, diff)
    mask = tf.logical_or(mask, temp_mask)
y = tf.boolean_mask(y,mask)

mean_x = tf.reduce_mean(x)
mean_y = tf.reduce_mean(y)

x_1 = x - mean_x
y_1 = y - mean_y

s_xy = tf.reduce_sum(tf.multiply(x_1,y_1))
s_x = tf.reduce_sum(x_1**2)
s_y = tf.reduce_sum(y_1**2)
sx_sy = tf.sqrt(tf.multiply(s_x,s_y))

p = tf.divide(s_xy, sx_sy)

negPearson_coeff = 1 - p
print(negPearson_coeff)