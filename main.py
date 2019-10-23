import model as m
from data_layer import DATA,Copula,RealData
import training
import numpy as np
import tensorflow as tf
from keras import backend as K
from metric import Test,copula_cmp
import numpy.linalg
import loss
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
def cmp_copula():
    models = m.Models(30,30,20,20,15,'selu',10)
    model = models.getModel()
    d = np.concatenate([lc[:tnum],rc[:tnum]],axis=1)
    cop = Copula(10*d)
    sim_data = cop.gendata(tnum)
    sim_data = np.array(sim_data)
    l_sim = sim_data[:,0:30]
    r_sim = sim_data[:,30:]
    left = np.concatenate([lc[:tnum],l_sim],axis=0)
    right = np.concatenate([rc[:tnum],r_sim],axis=0)
    left = left.tolist()
    right = right.tolist()
    val_left = lc[tnum:tnum+val]
    val_right = lc[tnum:tnum+val]

    test_left = lc[tnum+val:]
    test_right = rc[tnum+val:]
    
    training.train(left,right,val_left,val_right,300,30,model)
    test = Test(test_left,test_right,lu,ru_,lu_,ru,cov_mat,1, model)
    test.test_LtoR()

    model = models.getModel()
    training.train(lc[:tnum],rc[:tnum],val_left,val_right,300,30,model)
    test.model = model

    test.lessdata = 0
    test.test_LtoR()
    copula_cmp()

def cmp_copula2():
    data = DATA(30)
    num = 250
    tnum = int(num*0.6)
    val = int(num*0.2)
    lc,rc,lu,ru_,lu_,ru, cov_mat = data.get_data(num,100,100)
    
    models = m.Models(30,30,20,20,15,'selu',10)
    model = models.getModel()
    d = np.concatenate([lc[:tnum],rc[:tnum]],axis=1)
    cop = Copula(10*d)
    sim_data = cop.gendata(tnum)
    sim_data = np.array(sim_data)
    l_sim = sim_data[:,0:30]
    r_sim = sim_data[:,30:]
    left = np.concatenate([lc[:tnum],l_sim],axis=0)
    right = np.concatenate([rc[:tnum],r_sim],axis=0)
    left = left.tolist()
    right = right.tolist()
    val_left = lc[tnum:tnum+val]
    val_right = lc[tnum:tnum+val]

    test_left = lc[tnum+val:]
    test_right = rc[tnum+val:]
    
    #training.train_2(left,right,lu[:70],ru[:70],val_left,val_right,10,30,model)
    training.train(left,right,val_left,val_right,300,30,model)
    test = Test(test_left,test_right,lu[70:],ru_[70:],lu_[70:],ru[70:],cov_mat,2, model)
    test.test_LtoR()
    test.test_RtoL()

if __name__ == "__main__":
    data = RealData('G:\\Research\\DeepSavior\\DATA Base\\gtex6\\gtex-skin-wholeblood\\original\\Data\\',20)
    
    lc,rc = data.get_data()
    mer_data = np.concatenate([lc,rc],axis=1)
    #np.random.seed(100)
    #np.random.shuffle(mer_data)
    lc = mer_data[:,:len(lc[0])]
    rc = mer_data[:,len(lc[0]):]
    
    num = len(lc)
    tnum = int(num*0.8)
    val = int(num*0.2)

    model_dim = len(lc[0])
    print(num,model_dim)
    models = m.Models(model_dim, model_dim, int(model_dim/1), int(model_dim/1.2), int(model_dim/1.5), 'selu', .0001)
    model = models.getModel()
    d = np.concatenate([lc[:tnum],rc[:tnum]],axis=1)
    cop = Copula(d)
    sim_data = cop.gendata(10*tnum)
    sim_data = np.array(sim_data)
    l_sim = sim_data[:,0:model_dim]
    r_sim = sim_data[:,model_dim:]
    left = np.concatenate([lc[:tnum],l_sim],axis=0)
    right = np.concatenate([rc[:tnum],r_sim],axis=0)
    left = left.tolist()
    right = right.tolist()
    left = lc[:tnum]
    right = rc[:tnum]

    train_val_split = int(len(left)*0.7)
    val_left = left[train_val_split:]
    val_right = right[train_val_split:]

    left = left[:train_val_split] 
    right = right[:train_val_split]

    test_left = lc[tnum:]
    test_right = rc[tnum:]
    print(len(val_left), len(val_left[0]))
    print(len(test_left), len(test_left[0]))
    #training.train_2(left,right,lu[:70],ru[:70],val_left,val_right,10,30,model)
    lu=[]
    ru_=[]
    lu_=[]
    ru=[]
    cov_mat=cop.cov
    #fig, ax = plt.subplots(figsize=(50, 50),nrows=1, ncols=1)
    #ax.imshow(cov_mat, cmap='binary', interpolation='nearest')
    #fig.savefig('copula_test.png')
    training.train(left,right,val_left,val_right,50,30,model)
    
    test = Test(data.Common,data.Cleft[tnum:],data.Cright[tnum:],test_left,test_right,lu,ru_,lu_,ru,cov_mat,2, model)
    test.test_LtoR()
    #test.test_RtoL()
    '''test.test_U_LtoR()
    test.test_U_RtoL()'''


