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
        self.cov = np.cov(self.data.T)
        self.normal = stats.multivariate_normal([0 for i in range(len(data[0]))], self.cov)
        self.norm = stats.norm()
        self.var = []
        self.cdfs = []
        self.pdata = []
    def gendata(self,num):
        self.var = random.multivariate_normal([0 for i in range(len(self.cov[0]))], self.cov,num)
        self.cov = np.cov(self.var.T)
        for i in range(len(self.cov[0])):
          print(np.cov(self.var[:,i]),np.std(self.var[:,i]), np.cov(self.var[i,:]),math.sqrt(self.cov[i][i]) )
        #stds = [np.std(cop.var[:,j]) for j in range(len(self.cov[0]))]
        print(self.var.shape)
        for i in self.var:
            for j in range(len(i)):
                i[j]= i[j]/math.sqrt(self.cov[j][j])
        self.cdfs = self.norm.cdf(self.var)
        data = [ [ np.percentile(self.data[:,j],100*i[j]) for j in range(len(i))] for i in self.cdfs ]
        return data
def test_copula_2():
    Data = RealData('/media/ashiq/Education/Research/DeepSavior/DATA Base/gtex6/gtex-adipose-skin/original/Data/')
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

class transform():
    def __init__(self,num):
        self.pca = []
        self.std = []
        self.pcas = []
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

    def fit_get_both(self,left,right):
        mer_data = np.concatenate([left,right],axis=0)
        self.std = StandardScaler()
        self.std.fit(mer_data)
        ndata = self.std.transform(mer_data)
        self.pca = PCA(self.dim)
        self.pca.fit(ndata)
        ndata = self.pca.transform(ndata)

        self.pcas = StandardScaler()
        self.pcas.fit(ndata)
        ndata = self.pcas.transform(ndata)
        ndata = ndata +1

        return ndata[ :len(left), : ],ndata[ len(left): ,:]
    def get_both(self,left,right):
        left = self.std.transform(left)
        left = self.pca.transform(left)
        left = self.pcas.transform(left)
        left = left + 1

        right = self.std.transform(right)
        right = self.pca.transform(right)
        right = self.pcas.transform(right)
        right = right + 1

        return left,right

    def inv_trans(self, tdata):
        return self.std.inverse_transform(self.pca.inverse_transform(self.pcas.inverse_transform(tdata-1)))


class RealData():

    def __init__(self, path, Reduced_dim ):
        self.Cleft = []
        self.Cright = []
        self.Uleft = []
        self.Uright = []
        self.path = path
        self.Common = transform(Reduced_dim)
        
        

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
        tnum = int(len(self.Cleft)*0.8)
        return self.Cleft, self.Cright
        #tcl,tcr = self.Common.fit_get_both(self.Cleft[:tnum],self.Cright[:tnum])
        #tcl,tcr = self.Common.get_both(self.Cleft,self.Cright) 
        #return tcl.tolist(),tcr.tolist()
    

#test_copula_2()
'''data = [
    [1,2,3],
    [1.3,2.7,3.9],
    [.5,2.9,2.8],
    [3,1,2]
]   
print(data)
sdt = StandardScaler()
sdt.fit(data)
ndata = sdt.transform(data)

pca = PCA(1)
pca.fit(ndata)
pndata = pca.transform(ndata)
print(pndata)

pndata = pca.inverse_transform(pndata)
ndata = sdt.inverse_transform(pndata)
print(ndata)'''



