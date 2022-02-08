###### LOSS FUNCTIONS ###############
# Defines different Loss Functions. 
# Currently implemented:
# - Negative Pearson Coefficient

import tensorflow as tf


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
    
    negPearson_coeff = 1 - p

    return negPearson_coeff