from tensorflow.keras import  Model
from tensorflow.keras.layers import Input,Dense,concatenate,BatchNormalization,Dropout,Add
from tensorflow.keras import backend as K,activations
from tensorflow import Tensor as T
from tensorflow.python.keras.layers import Layer
import numpy as np


import loss
#hdim_deep=10
#hdim_deep2=20
commonx= None
commony = None
def gaussian(x):
     return  K.pow(x,3)
    #return 2*K.exp(-1*K.pow(x,2))+x

class ZeroPadding(Layer):
     def __init__(self, **kwargs):
          super(ZeroPadding, self).__init__(**kwargs)

     def call(self, x, mask=None):
          return K.zeros_like(x)

     def get_output_shape_for(self, input_shape):
          return input_shape

class CorrnetCost(Layer):
     def __init__(self,lamda, **kwargs):
          super(CorrnetCost, self).__init__(**kwargs)
          self.lamda = lamda
     def cor(self,y1, y2,left,right, lamda):

          y1_mean = K.mean(y1, axis=0)
          y1_centered = y1 - y1_mean
          y2_mean = K.mean(y2, axis=0)
          y2_centered = y2 - y2_mean
          corr_nr = K.sum(y1_centered * y2_centered, axis=0)
          corr_dr1 = K.sqrt(K.sum(y1_centered * y1_centered, axis=0) + 1e-8)
          corr_dr2 = K.sqrt(K.sum(y2_centered * y2_centered, axis=0) + 1e-8)
          corr_dr = corr_dr1 * corr_dr2
          corr = corr_nr / corr_dr

          return K.sum(corr) * lamda

     def call(self ,x ,mask=None):
          hx,hy, left, right=x[0], x[1], x[2], x[3]
          corr = self.cor(hx,hy,left,right,self.lamda)
          return corr
     def get_output_shape_for(self, input_shape):
        print(input_shape[0][0])
        return (input_shape[0][0],input_shape[0][1])


def corr_loss(y_true, y_pred):
        return y_pred
        #return tf.fill(y_pred.get_shape(), 9.9)

class Models:

          def __init__(self, dim_lef, dim_rig, layer1, layer2, common, nonlin,last_layer_act, lamda):
               global commonx
               global commony
               self.inputDimx = dim_lef
               self.inputDimy = dim_rig
               self.hdim_deep = layer1
               self.hdim_deep2 = layer2
               self.dim_common = common
               commonx = common
               commony = common
               self.Loss = loss.mylosses(lamda,dim_lef)
               self.nonlin = nonlin
               self.last_layer_act = last_layer_act
               self.lamda = lamda



          def getModel(self):
               inpx = Input(shape=(self.inputDimx,))
               inpy = Input(shape=(self.inputDimy,))


               hl = Dense(self.hdim_deep,activation=self.nonlin)(inpx)
               #hl = BatchNormalization()(hl)
               #hl = Dropout(0.3)(hl)
               hl = Dense(self.hdim_deep2, activation=self.nonlin,name='hid_l2')(hl)
               #hl = Dropout(0.3)(hl)
               #hl = BatchNormalization()(hl)
               #hl = Dense(self.hdim_deep2, activation=self.nonlin,name='hid_l1')(hl)
               #hl = BatchNormalization()(hl)
               hl = Dense(self.dim_common, activation=self.nonlin,name='hid_l')(hl)
               #hl = BatchNormalization()(hl)


               hr = Dense(self.hdim_deep,activation=self.nonlin)(inpy)
               #hr = Dropout(0.3)(hr)
               #hr = BatchNormalization()(hr)
               hr = Dense(self.hdim_deep2, activation=self.nonlin,name='hid_r2')(hr)
               #hr = BatchNormalization()(hr)
               #hr = Dense(self.hdim_deep2, activation=self.nonlin,name='hid_r1')(hr)
               #hr = Dropout(0.3)(hr)
               #hr = BatchNormalization()(hr)
               hr = Dense(self.dim_common, activation=self.nonlin,name='hid_r')(hr)
               #hr = BatchNormalization()(hr)


               h =  Add()([hl,hr])


               recx = Dense(self.hdim_deep2,activation=self.nonlin)(h)
               #recx = Dropout(0.3)(recx)
               #recx = BatchNormalization()(recx)
               recx = Dense(self.hdim_deep2,activation=self.nonlin)(recx)
               #recx = BatchNormalization()(recx)
               #recx = Dense(self.hdim_deep,activation=gaussian)(recx)
               #recx = Dropout(0.3)(recx)
               #recx = BatchNormalization()(recx)
               recx = Dense(self.inputDimx,activation=self.last_layer_act)(recx)
               #
               recy = Dense(self.hdim_deep2,activation=self.nonlin)(h)
               #recy = Dropout(0.3)(recy)
               #recy = BatchNormalization()(recy)
               recy = Dense(self.hdim_deep2,activation=self.nonlin)(recy)
               #recy = BatchNormalization()(recy)
               #recy = Dense(self.hdim_deep,activation=gaussian)(recy)
               #recy = Dropout(0.3)(recy)
               #recy = BatchNormalization()(recy)
               recy = Dense(self.inputDimy,activation=self.last_layer_act)(recy)
               #bhout = concatenate([recx,recy,h])

               branchModel = Model( [inpx,inpy],[recx,recy,h])
               branchModel.summary()

               [recx1,recy1,h1] = branchModel( [inpx, ZeroPadding()(inpy)])
               [recx2,recy2,h2] = branchModel( [ZeroPadding()(inpx), inpy ])

               # reconstruction from combined view1 and view2
               [recx3,recy3,h] =branchModel([inpx, inpy])


               #corr = CorrnetCost(-0.20)([h1,h2,inpx, inpy])
               H= concatenate([h1,h2,inpx,inpy])
               sml =  Model( [inpx,inpy],recx)
               smr = Model( [inpx,inpy],recy)

               model = Model( [inpx,inpy],[recx1,recx2,recx3,recy1,recy2,recy3,H])
               model.compile( loss=[self.Loss.square_loss, self.Loss.square_loss,
               self.Loss.square_loss,
               self.Loss.square_loss,self.Loss.square_loss,
               self.Loss.square_loss,self.Loss.correlationLoss],optimizer="adam")
               #model.summary()
               return model, sml, smr

          def getModel_2(self):
               inpx = Input(shape=(self.inputDimx,))
               inpy = Input(shape=(self.inputDimy,))


               hl = Dense(self.hdim_deep,activation=self.nonlin)(inpx)
               hl = Dense(self.hdim_deep2, activation=self.nonlin,name='hid_l1')(hl)
               hl = Dense(self.dim_common, activation=self.nonlin,name='hid_l')(hl)


               hr = Dense(self.hdim_deep,activation=self.nonlin)(inpy)
               hr = Dense(self.hdim_deep2, activation=self.nonlin,name='hid_r1')(hr)
               hr = Dense(self.dim_common, activation=self.nonlin,name='hid_r')(hr)


               h =  Merge(mode="sum")([hl,hr])


               recx = Dense(self.hdim_deep2,activation=self.nonlin)(h)
               recx = Dense(self.hdim_deep,activation=self.nonlin)(h)
               recx = Dense(self.inputDimx,activation='selu')(h)
               #
               recy = Dense(self.hdim_deep2,activation=self.nonlin)(h)
               recy = Dense(self.hdim_deep,activation=self.nonlin)(h)
               recy = Dense(self.inputDimy,activation='selu')(h)
               #bhout = concatenate([recx,recy,h])

               branchModel = Model( [inpx,inpy],[recx,recy,h])
               branchModel.summary()

               [recx1,recy1,h1] = branchModel( [inpx, ZeroPadding()(inpy)])
               [recx2,recy2,h2] = branchModel( [ZeroPadding()(inpx), inpy ])

               # reconstruction from combined view1 and view2
               [recx3,recy3,h] =branchModel([inpx, inpy])


               corr = CorrnetCost(-1*self.lamda)([h1,h2,inpx, inpy])
               #H= concatenate([h1,h2,inpx,inpy])

               model = Model( [inpx,inpy],[recx1,recx2,recx3,recy1,recy2,recy3,corr])
               model.compile( loss=[self.Loss.square_loss, self.Loss.square_loss,
               self.Loss.square_loss,
               self.Loss.square_loss,self.Loss.square_loss,
               self.Loss.square_loss,corr_loss],optimizer="adam")
               #model.summary()
               return model
