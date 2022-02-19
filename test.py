from cv2 import threshold
import numpy as np
import tensorflow as tf
import h5py

import heartpy as hp

a = [2,28,132,163,174,196, 48,70,91,112]
b = [3, 6,112, 150, 180, 160, 28, 50, 91]


x = tf.convert_to_tensor(a, dtype=tf.float64)
y = tf.convert_to_tensor(b, dtype=tf.float64)


threshold = 50
data_reshaped = tf.reshape(y, (1, -1, 1)) # (1, N, 1)
a = tf.nn.max_pool(data_reshaped, (20,), 1,'SAME')
max_pooled_in_tensor = tf.nn.pool(data_reshaped, window_shape=(20,), pooling_type='MAX', padding='SAME')
maxima = tf.equal(data_reshaped,max_pooled_in_tensor) # (1, N, 1)
def fn(t):

    return 1


r = tf.greater_equal(data_reshaped,(max_pooled_in_tensor - threshold))
r = tf.squeeze(r)
test = tf.map_fn(fn=lambda t: fn(t), elements=r)

tf.no_gradient()

maxima = tf.cast(maxima, tf.float64)
maxima = tf.squeeze(maxima) # (N,1)
a = tf.map_fn(fn=lambda t: tf.range(t, t + 3), elems=tf.constant([3, 5, 2]))
def fn(t):
    a = tf.range(t,t+2)
    return a
maxima = tf.map_fn(fn=lambda t: fn(t), elems=maxima)
#range = tf.where(tf.squeeze(max_pooled_in_tensor))
range = tf.reshape(range, (-1,))

#t = tf.boolean_mask(range, maxima)
#v = tf.range(0, tf.size(y))
#peak = tf.math.top_k(maxima, k=2)

#maxima = tf.cast(maxima, tf.bool)



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
val = x
val = tf.boolean_mask(val, mask)

def fn(item):
    def true_fn():
        return min
    def false_fn():
        return tf.cast(-1, tf.float64)
    diff = tf.abs(x - item) # diff of truth data and item
    min = tf.reduce_min(diff) # minimum of diff
    min = tf.cond(tf.less(min, max_offset), true_fn, false_fn)
    temp_mask = tf.equal(min, diff)
    return temp_mask
mask1 = tf.map_fn(fn=lambda item: fn(item), elems=y, fn_output_signature=tf.bool)
mask1 = tf.reduce_any(mask1, 0)
x = tf.boolean_mask(x,mask1)

# # check if outliners are in pred
# mask = tf.cast(tf.zeros(tf.size(y)), tf.bool)
# for item in x: 
#     diff = tf.abs(y - item) # diff of truth data and item
#     min = tf.reduce_min(diff) # minimum of diff
#     min = tf.cond(tf.less(min, max_offset), true_fn, false_fn)
#     temp_mask = tf.equal(min, diff)
#     mask = tf.logical_or(mask, temp_mask)
def fn2(item):
        def true_fn():
            return min
        def false_fn():
            return -1
        diff = tf.abs(y - item) # diff of truth data and item
        min = tf.reduce_min(diff) # minimum of diff
        min = tf.cond(tf.less(min, max_offset), true_fn, false_fn)
        temp_mask = tf.equal(min, diff)
        return temp_mask
mask2 = tf.map_fn(fn=lambda item: fn2(item), elems=x, fn_output_signature=tf.bool)
mask2 = tf.reduce_any(mask2, 0)
y = tf.boolean_mask(y,mask2)

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