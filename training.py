import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model
import loss
def train( X_train_l , X_train_r ,val_left, val_right, epoch,batch_size, model ):
    X_train_l = np.array(X_train_l)
    X_train_r = np.array(X_train_r)
    val_left = np.array(val_left)
    val_right = np.array(val_right)
    checkpointer = ModelCheckpoint('model.h5', verbose=0, save_best_only=True)
    '''
    validation_data=([val_left,val_right],[val_left,val_left,val_left,val_right,val_right,val_right,np.ones((val_left.shape[0],val_left.shape[1]))]),
                shuffle=True,
                  callbacks=[
                  checkpointer,
              ]'''
    model.fit([X_train_l,X_train_r], [X_train_l,X_train_l,X_train_l,X_train_r,X_train_r,X_train_r,np.ones((X_train_l.shape[0],X_train_l.shape[1]))],
               nb_epoch=epoch,batch_size=batch_size,verbose=0,
               validation_data=([val_left,val_right],[val_left,val_left,val_left,val_right,val_right,val_right,np.ones((val_left.shape[0],val_left.shape[1]))]),
                shuffle=True,
                  callbacks=[
                  checkpointer,
              ]
               
               )
    model.load_weights('model.h5')
def train_2(left_c, right_c, l_u, r_u,val_left,val_right, epoch,batch_size, model):
    l_c = np.array(left_c)
    r_c = np.array(right_c)
    r_u_dum = [[0 for i in range(len(l_u[0]))] for j in range(len(l_u))]
    l_u_dum = [[0 for i in range(len(r_u[0]))] for j in range(len(r_u))]
    l_u = np.array(l_u)
    r_u = np.array(r_u)
    l_u_dum = np.array(l_u_dum,)
    r_u_dum = np.array(r_u_dum)
    L_c = np.copy(l_c)
    R_c = np.copy(r_c)
    val_left = np.array(val_left)
    val_right = np.array(val_right)
    checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    model.fit( [ L_c, R_c ], [L_c,L_c,L_c, R_c, R_c, R_c,np.ones((L_c.shape[0],L_c.shape[1]))],
      batch_size=batch_size, epochs=epoch,
               validation_data=([val_left,val_right],[val_left,val_left,val_left,val_right,val_right,val_right,np.ones((val_left.shape[0],val_left.shape[1]))]),
                shuffle=True,
                  callbacks=[
                  checkpointer,
              ])
    model.load_weights('model.h5')
    for i  in range(5):
      model.fit( [ L_c, R_c ], [L_c,L_c,L_c, R_c, R_c, R_c,np.ones((L_c.shape[0],L_c.shape[1]))],
      batch_size=batch_size, epochs=int(epoch/5),
               validation_data=([val_left,val_right],[val_left,val_left,val_left,val_right,val_right,val_right,np.ones((val_left.shape[0],val_left.shape[1]))]),
                shuffle=True,
                  callbacks=[
                  checkpointer,
              ])
      model.load_weights('model.h5')
      _,_,_,_,_,temp_right,_ = model.predict([l_u,r_u_dum])
      _,_,temp_left,_,_,_,_ = model.predict([l_u_dum,r_u])

      L_c = np.concatenate((l_c,l_u,temp_left),axis=0)
      R_c = np.concatenate((r_c,temp_right,r_u),axis=0)
    model.load_weights('model.h5')




  
