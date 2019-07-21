from keras import backend as K
import tensorflow as tf
import keras.losses as ls
import model
import math
#import tensorflow.math
#def loss_sup(x):
    #return (2-x)*(2-x)
class mylosses():
    def __init__(self, lamda, dim):
        self.Lambda = 1*lamda
        self.dim = dim

    def correlationLoss(self, fake, H):
        y1 = H[:,:model.commonx]
        y2 = H[:,model.commonx:2*model.commonx]
        left = H[:,2*model.commonx:2*model.commonx+self.dim]
        right = H[:,2*model.commonx+self.dim:]
        y1_mean = K.mean(y1, axis=0)
        y1_centered = y1 - y1_mean
        y2_mean = K.mean(y2, axis=0)
        y2_centered = y2 - y2_mean
        corr_nr = K.sum(y1_centered * y2_centered, axis=0)
        corr_dr1 = K.sqrt(K.sum(y1_centered * y1_centered, axis=0) + 1e-8)
        corr_dr2 = K.sqrt(K.sum(y2_centered * y2_centered, axis=0) + 1e-8)
        corr_dr = corr_dr1 * corr_dr2
        corr = corr_nr / corr_dr
        function_to_map = lambda x: (1-x)*(1-x)  # Where `f` instantiates myCustomOp.
        corr = tf.map_fn(function_to_map, corr)
        return K.sum(corr) * self.Lambda

    def square_loss(self, y_true, y_pred):
        error = ls.mean_squared_error(y_true,y_pred)
        return error