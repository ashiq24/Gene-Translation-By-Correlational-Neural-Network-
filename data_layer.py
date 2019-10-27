from numpy import random
from random import sample
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import math
plt.rcParams.update({'font.size': 22})
class DATA():
    def __init__(self, Num_feachers):
        self.feachers = Num_feachers
    def get_covmat(self):
        return datasets.make_spd_matrix(2*self.feachers)
        #return datasets.make_spd_matrix(2*self.feachers,random_state=100)
    

    def get_means(self):
        means = []
        for i in range(2*self.feachers):
            means.append(random.uniform(0, 10))

        return np.array(means)


    def get_data(self, common,l_unique,r_unique):
        total = common + l_unique+r_unique
        cov_mat = self.get_covmat()
        mean_vec1 = self.get_means()
        data_bulk = random.multivariate_normal(mean_vec1,cov_mat,int(total/2))
        mean_vec2 = self.get_means()
        data_bulk_2 = random.multivariate_normal(mean_vec2,cov_mat,total-int(total/2))
        data_bulk = np.concatenate([data_bulk,data_bulk_2],axis=0)
        
        mean_vec1 = mean_vec1.reshape( (1,len(mean_vec1)))
        mean_vec2 = mean_vec2.reshape( (1,len(mean_vec2)))
       

        cov_mat = cov_mat + (0.5*np.matmul(mean_vec1.T,mean_vec1)+0.5*np.matmul(mean_vec2.T,mean_vec2)) - np.matmul((0.5*mean_vec1+0.5*mean_vec2).T,(0.5*mean_vec1+0.5*mean_vec2))
        
        print(len(data_bulk), len(data_bulk[0]))
        random.shuffle(data_bulk)
        #plt.boxplot(data_bulk)
       
        l_com = [i[:self.feachers] for i in data_bulk[:common]]
        r_com = [i[self.feachers:] for i in data_bulk[:common]]
        l_u = [i[:self.feachers] for i in data_bulk[common:common+l_unique]]
        r_u_ = [i[self.feachers:] for i in data_bulk[common:common+l_unique]]
        l_u_ = [i[:self.feachers] for i in data_bulk[len(data_bulk)-r_unique:]]
        r_u = [i[self.feachers:] for i in data_bulk[len(data_bulk)-r_unique:]]
        
        df = pd.DataFrame(l_com, columns=[str(i) for i in range(self.feachers)] )
        df.plot.box(patch_artist=True)
        plt.savefig('leftinput_com.jpg')
        df1 = pd.DataFrame(r_com, columns=[str(i) for i in range(self.feachers)] )
        df1.plot.box(patch_artist=True)
        plt.savefig('rightinput_com.jpg')
        
        return l_com,r_com,l_u,r_u_,l_u_,r_u,cov_mat
class Copula():
    def __init__(self,data):
        self.data = np.array(data)
        if(len(data)<2):
            raise  Exception('input data must have multiple samples')

        self.cov = np.cov(self.data.T)
        self.normal = stats.multivariate_normal([0 for i in range(len(data[0]))], self.cov,allow_singular=True)
        self.norm = stats.norm()
        self.var = []
        self.cdfs = []
        self.pdata = []
    def gendata(self,num):
        self.var = random.multivariate_normal([0 for i in range(len(self.cov[0]))], self.cov,num)
        self.cov = np.cov(self.var.T)
        #for i in range(len(self.cov[0])):
        #print(np.cov(self.var[:,i]),np.std(self.var[:,i]), np.cov(self.var[i,:]),math.sqrt(self.cov[i][i]) )
        #stds = [np.std(cop.var[:,j]) for j in range(len(self.cov[0]))]
        print(self.var.shape)
        for i in self.var:
            for j in range(len(i)):
                i[j]= i[j]/math.sqrt(self.cov[j][j])
        self.cdfs = self.norm.cdf(self.var)
        data = [ [ np.percentile(self.data[:,j],100*i[j]) for j in range(len(i))] for i in self.cdfs ]
        return data


class transform():
    def __init__(self,num):
        self.pca_l = None
        self.std_l = None
        self.std2_l = None
        self.pca_r = None
        self.std_r = None
        self.std2_r = None
        self.dim = num

    def fit_get(self,data):
        self.std = StandardScaler()
        self.std.fit(data)
        ndata = self.std.transform(data)
        self.pca = PCA(self.dim)
        self.pca.fit(ndata)
        ndata = self.pca.transform(ndata)
        nstd = StandardScaler()
        nstd.fit(ndata)
        ndata = nstd.transform(ndata)
        print(ndata[:][1])
        ndata = ndata + 1
        return ndata

    def fit_get_both(self,left,right, uleft, uright):
        ndata = np.concatenate([left,right], axis=0)
        self.std_l = StandardScaler(with_std=True)
        self.std_l.fit(ndata)
        ndata = self.std_l.transform(ndata)
        self.pca_l = PCA(self.dim)
        self.pca_l.fit(ndata)
        ndata = self.pca_l.transform(ndata)

        self.std2_l = StandardScaler()
        self.std2_l.fit(ndata)
        ndata = self.std2_l.transform(ndata)
        #new_l = ndata +1
        #right
        ndata = np.concatenate([left,right], axis=0)
        self.std_r = StandardScaler(with_std=True)
        self.std_r.fit(ndata)
        ndata = self.std_r.transform(ndata)
        self.pca_r = PCA(self.dim)
        self.pca_r.fit(ndata)
        ndata = self.pca_r.transform(ndata)

        self.std2_r = StandardScaler(with_std=True)
        self.std2_r.fit(ndata)
        ndata = self.std2_r.transform(ndata)
        #new_r = ndata + 1

        return None, None

    def get_both(self,left,right, Uleft, Uright):
        T_left = self.std_l.transform(left)
        T_left = self.pca_l.transform(T_left)
        T_left = self.std2_l.transform(T_left)
        #T_left = T_left

        T_right = self.std_r.transform(right)
        T_right = self.pca_r.transform(T_right)
        T_right = self.std2_r.transform(T_right)
        #T_right = T_right

        T_Uleft = self.std_l.transform(Uleft)
        T_Uleft = self.pca_l.transform(T_Uleft)
        T_Uleft = self.std2_l.transform(T_Uleft)
        #T_Uleft = T_Uleft

        T_Uright = self.std_r.transform(Uright)
        T_Uright = self.pca_r.transform(T_Uright)
        T_Uright = self.std2_r.transform(T_Uright)
        #T_Uright = T_Uright

        return T_left, T_right, T_Uleft, T_Uright

    def inv_trans_left(self, T_left):
        #return self.std_l.inverse_transform(self.pca_l.inverse_transform(T_left))
        return self.std_l.inverse_transform(self.pca_l.inverse_transform(self.std2_l.inverse_transform(T_left)))
    def inv_trans_right(self, T_right):
        #return self.std_r.inverse_transform(self.pca_r.inverse_transform(T_right))
        return self.std_r.inverse_transform(self.pca_r.inverse_transform(self.std2_r.inverse_transform(T_right)))


class RealData():

    def __init__(self, path, Reduced_dim ):
        self.Cleft = []
        self.Cright = []
        self.Uleft = []
        self.Uright = []
        self.path = path
        self.Transform = transform(Reduced_dim)
        
        

    def load(self):
        data = pd.read_csv(self.path+'exps_CommonLeft.csv')
        data = data.drop(columns=['Unnamed: 0'])
        self.Cleft = data.values
        data = pd.read_csv(self.path+'exps_commonRight.csv')
        data = data.drop(columns=['Unnamed: 0'])
        self.Cright = data.values

        '''mer_data = np.concatenate([self.Cleft,self.Cright],axis=1)
        np.random.seed(100)
        np.random.shuffle(mer_data)
        self.Cleft = mer_data[:,:len(self.Cleft[0])]
        self.Cright = mer_data[:,len(self.Cleft[0]):]'''

        data = pd.read_csv(self.path+'exps_UniqLeft.csv')
        data = data.drop(columns=['Unnamed: 0'])
        self.Uleft = data.values
        data = pd.read_csv(self.path+'exps_UniqRight.csv')
        data = data.drop(columns=['Unnamed: 0'])
        self.Uright = data.values

    def get_data(self,):
        self.load()
        tnum = int(len(self.Cleft)*0.7)
        tcl,tcr = self.Transform.fit_get_both(self.Cleft[:tnum],self.Cright[:tnum],self.Uleft, self.Uright)
        tcl,tcr,tul,tur = self.Transform.get_both(self.Cleft,self.Cright,self.Uleft,self.Uright)
        return tcl.tolist(),tcr.tolist(), tul.tolist(), tur.tolist()
    
def test_copula_2():
    Data = RealData('E:\\Data')
    lc,rc = Data.get_data()
    data = np.concatenate([lc,rc],axis=1)
    cop = Copula(data)
    cov_mat = cop.cov
    num=len(lc)
    pdata = cop.gendata(num)
    npdata = np.array(pdata)
    cop_cov = np.cov(npdata.T)
    fig, ax = plt.subplots(figsize=(20, 20),nrows=1, ncols=2)
    ax[0].imshow(cov_mat, cmap='binary', interpolation='nearest')
    ax[1].imshow(cop_cov, cmap='binary', interpolation='nearest')
    ax[0].set_title('real cov')
    ax[1].set_title('copula cov')
    fig.savefig('copula_test.png')

    from metric import Test

    t = Test([],[],[],[],[],[],data,[],[])
    
    t.gengraph(data, pdata,'data','gen_data')




