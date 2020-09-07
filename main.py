import model as m
from data_layer import DATA,Copula,RealData
import training
import numpy as np
from os import listdir
from metric import Test,copula_cmp,boxplot
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import gc
from keras import backend as K
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



def impute(path):
    filepath = path  # sys.argv[1]
    pca_com = 25  # int(sys.argv[2])
    copula_num = 6000  # int(sys.argv[3])
    activation = 'selu'  # sys.argv[4]
    activation_last_layer = 'linear'  # sys.argv[4]
    train_op = 1  # int(sys.argv[6])
    test_op = 1  # int(sys.argv[7])
    lamda = 20  # float(sys.argv[8])

    data = RealData(filepath, pca_com)

    lc, rc, lu, ru = data.get_data()

    # suffeling the data
    '''
    mer_data = np.concatenate([lc,rc],axis=1)
    np.random.seed(100)
    np.random.shuffle(mer_data)

    lc = mer_data[:,:len(lc[0])]
    rc = mer_data[:,len(lc[0]):]
    '''
    num = len(lc)
    print("number of samples ", num)
    tnum = int(num * 0.7)
    test = int(num * 0.3)
    print("training", tnum, " testing", test)

    model_dim = len(lc[0])
    print(num, model_dim)
    '''
    Models(1st_layer_left, 1st_layer_right, 2nd_layer, 3rd_layer, common_layer)
    '''
    models = m.Models(model_dim, model_dim, int(model_dim / 1.5), int(model_dim / 1.8), int(model_dim / 2.5),
                      activation, activation_last_layer, lamda)
    model = models.getModel()
    if (copula_num > 0):
        d = np.concatenate([lc[:tnum], rc[:tnum]], axis=1)
        cop = Copula(d)
        sim_data = cop.gendata(copula_num)
        sim_data = np.array(sim_data)
        l_sim = sim_data[:, 0:model_dim]
        r_sim = sim_data[:, model_dim:]
        left = np.concatenate([lc[:tnum], l_sim], axis=0)
        right = np.concatenate([rc[:tnum], r_sim], axis=0)
        left = left.tolist()
        right = right.tolist()

    else:
        left = lc[:tnum]
        right = rc[:tnum]

    train_val_split = int(len(left) * 0.7)
    val_left = left[train_val_split:]
    val_right = right[train_val_split:]

    left = left[:train_val_split]
    right = right[:train_val_split]

    test_left = lc[tnum:]
    test_right = rc[tnum:]
    print(len(val_left), len(val_left[0]))
    print(len(test_left), len(test_left[0]))
    '''
    train(train_left, train_right, val_left, val_right, epochs, batch_size, model)
    '''
    if (train_op == 1):
        training.train(left, right, val_left, val_right, 400, 100, model)
    else:
        training.train_2(left, right, lu, ru, val_left, val_right, 10, 30, model)

    ru_ = []
    lu_ = []

    cov_mat = None
    # fig, ax = plt.subplots(figsize=(50, 50),nrows=1, ncols=1)
    # ax.imshow(cov_mat, cmap='binary', interpolation='nearest')
    # fig.savefig('copula_test.png')

    test = Test(data.Transform, data.Cleft[tnum:], data.Cright[tnum:], test_left, test_right, lu, ru_, lu_, ru, cov_mat,
                2, model)
    pc, sc, ks, sql = test.test_LtoR()
    del model

    del cop
    del left
    del right

    return  num, pc, sc, ks, sql
    #if test_op == 1:
        #test.test_LtoR()
    #else:
        #test.test_RtoL()
    #test.test_LtoL();
    '''test.test_U_LtoR()
    test.test_U_RtoL()'''
if __name__ == "__main__":
    a = []
    b = []
    #boxplot(a,b, 'test')
    #exit
    names  = []
    PC = []
    SC = []
    KS = []
    SQL = []
    NUM = []
    counter = 0
    for i in listdir("E:\All_Data"):

        _, l, r, _ = i.split('-')
        if l==r or 'WholeBlood'!=l:
            continue
        print("working with tissue", counter, l , r)

        try:
            num, pc, sc, ks, sql = impute("E:\All_Data\\"+i+'\\')
            PC.append(pc)
            SC.append(sc)
            KS.append(ks)
            SQL.append(sql)
            NUM.append(num)
            names.append(l+"-"+r)
            counter+=1
            gc.collect()
        except :
            print(i, "error occured")


    import pickle


    with open('exp_data.pickle', 'wb') as handle:
        pickle.dump(names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(PC, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(SC, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(NUM, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(SQL, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(KS, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(names)
    boxplot(PC,names,"pearson")
    #boxplot(KS,names,"KS_TEST")






