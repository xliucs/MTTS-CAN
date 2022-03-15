###### LOSS FUNCTIONS ###############
# Defines different Loss Functions. 
# Currently implemented:
# - Negative Pearson Coefficient

from tkinter.tix import Y_REGION
import tensorflow as tf

import tensorflow.keras.backend as K

# Negative Pearson Coefficient
# x: truth rPPG    y: predicted rPPG
def negPearsonLoss(x,y):
    mean_x = tf.reduce_mean(x)
    mean_y = tf.reduce_mean(y)

    x_1 = x - mean_x
    y_1 = y - mean_y

    s_xy = tf.reduce_sum(tf.multiply(x_1,y_1))
    s_x = tf.reduce_sum(x_1**2)
    s_y = tf.reduce_sum(y_1**2)
    sx_sy = tf.sqrt(tf.multiply(s_x,s_y))
    
    p = tf.divide(s_xy, sx_sy)
    
    negPearson_coeff = 1. - p

    return negPearson_coeff


def gaussian_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, (-1,))
    y_true = tf.reshape(y_true, (-1,))
    return -tf.reduce_sum(y_true*y_pred)

def time_error_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, (-1,))
    y_true = tf.reshape(y_true, (-1,))
    return tf.reduce_sum(y_true*y_pred)

def MRPE_parameter_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))
    AE = tf.abs(y_true-y_pred)/y_true
    return tf.reduce_mean(AE)

def negPearsonLoss_onlyPeaks(y_true, y_pred):
    peaks_true = get_peaks(y_true)
    peaks_pred = get_peaks(y_pred)
    peaks_pred = peaks_pred[0:10]
    peaks_true = peaks_true[0:10]
    #peaks_true, peaks_pred = filt_peaks(peaks_true, peaks_pred)
    peaks_true = tf.cast(peaks_true, tf.float32)
    peaks_pred = tf.cast(peaks_pred, tf.float32)

    negPeaLoss = negPearsonLoss(peaks_true, peaks_pred)
    return negPeaLoss


def get_peaks(y):
    # y: (N,)
    data_reshaped = tf.reshape(y, (1, -1, 1)) # (1, N, 1)
    max_pooled_in_tensor =  tf.nn.max_pool(data_reshaped, (20,), 1,'SAME')
    
    #maxima = tf.stop_gradient(tf.equal(data_reshaped,max_pooled_in_tensor)) # (1, N, 1)
    #maxima = tf.cast(maxima, tf.float32)
    #maxima = tf.squeeze(maxima) # (N,1)
    #peaks = tf.where(maxima, name="Where") # now only the Peak Indices (A, 3)
    #tf.no_gradient("Where")
    #peaks = tf.reshape(peaks, (-1,)) # (A,1)

    return max_pooled_in_tensor

# x: true y: prediction
# input: peaks of truth and prediction as tensor...
@tf.function
def filt_peaks(x,y):
    def true_fn():
        return min
    def false_fn():
        return tf.cast(-1, tf.int64)
    max_offset = 10
    mask = tf.cast(tf.zeros(tf.size(x)),tf.bool) # tensor with size of x (truth data)
    # check which peaks of truth are recognized in pred
    min = 0
    min = tf.cast(min, tf.int64)

    def fn(item):
        diff = tf.abs(x - item) # diff of truth data and item
        min = tf.reduce_min(diff) # minimum of diff
        min = tf.cond(tf.less(min, max_offset), true_fn, false_fn)
        temp_mask = tf.equal(min, diff)
        mask = tf.logical_or(mask, temp_mask)
        return mask
    mask = tf.map_fn(fn=lambda item: fn(item), elems=y)
    # for item in y: # items of predicion
    #     diff = tf.abs(x - item) # diff of truth data and item
    #     min = tf.reduce_min(diff) # minimum of diff
    #     min = tf.cond(tf.less(min, max_offset), true_fn, false_fn)
    #     temp_mask = tf.equal(min, diff)
    #     mask = tf.logical_or(mask, temp_mask)

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
    return x, y 