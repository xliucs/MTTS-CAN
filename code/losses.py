###### LOSS FUNCTIONS ###############
# Defines different Loss Functions. 
# Currently implemented:
# - Negative Pearson Coefficient
from tkinter import Y
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


# Negative Pearson Coefficient
# x: truth rPPG    y: predicted rPPG
def negPearsonLoss(x, y):
    T = y.shape[0]
    if T == None:
        T = 1
    print(x)
    print(y)
    print(y.shape)
    print("T: ", T)
    
    numerator = T* tf.reduce_sum(y*x) - tf.reduce_sum(y)*tf.reduce_sum(x)
    d_coeff1 = T* tf.reduce_sum(y**2) - tf.reduce_sum(y)**2
    d_coeff2 = T*tf.reduce_sum(x**2) - tf.reduce_sum(x)**2

    pear_coeff = numerator/(tf.sqrt(d_coeff1*d_coeff2))

    negPearson_coeff = 1 - pear_coeff

    return negPearson_coeff 