import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr, pearsonr, ks_2samp
import matplotlib.backends.backend_pdf as mpdf
import pickle
import pandas as pd
plt.rcParams.update({'font.size': 10})
lowdata = []
moredata = []
class Test():
    def __init__(self,transformation,real_lc, real_rc, l_c, r_c, l_u, r_u_, l_u_, r_u, cov_mat,ismoredata, model):
        self.transformation = transformation
        self.real_lc = real_lc
        self.real_rc = real_rc
        self.l_c = l_c
        self.r_c = r_c
        self.l_u = l_u
        self.r_u_ = r_u_
        self.l_u_ = l_u_
        self.r_u = r_u
        self.cov_mat = cov_mat
        self.model = model
        self.lessdata = ismoredata
    def discriminate(self, ks_score, inp_name, ter_name):
        cov_sum = []
        print('here')
        l = len(ks_score)
        if inp_name == 'left':
            for i in range(l,2*l,1):
                cov_sum.append(sum( [j for j in self.cov_mat[i]] )/l)
        else:
            for i in range(0,l,1):
                cov_sum.append(sum( [j for j in self.cov_mat[i]] )/l)
        fig, ax = plt.subplots(figsize=(25, 20),nrows=1, ncols=1,sharey=True)
        ax.scatter(ks_score,cov_sum)
        
        plt.xlabel("Co_relation")
        plt.ylabel("cov_sum ")
        plt.title("Features Check")
        fig.savefig(inp_name+' to '+ter_name+'.jpg')


    def gengraph_bigdata(self,pred, target, inp_name, ter_name):
        pdf = mpdf.PdfPages(inp_name+"  "+ter_name+'.pdf')
        pearson_peo = []
        square_error_peo = []

        for i in range(len(pred)):
            #print(pred[i][0:20])
            #print(target[i][0:20])
            p, _ = pearsonr(pred[i],target[i])
            pearson_peo.append(p)
            square_error_peo.append( (np.square(np.array(pred[i]) - np.array(target[i]))).mean(axis=0) )
        
        pearson = []
        spearman = []
        square_error = []
        Ks_test = []

        for k in range(len(target[0])-1):
            
            fpred = [i[k] for i in pred]
            frael = [i[k] for i in target]
            mn = min(min(fpred),min(frael))-5
            mx = max(max(fpred),max(frael))+5
            p, _ = pearsonr(fpred,frael)
            s,_ = spearmanr(fpred,frael)
            pearson.append( p )
            spearman.append( s)
            square_error.append( (np.square(np.array(fpred) - np.array(frael))).mean(axis=0) )
            ks,_ = ks_2samp(fpred,frael)
            Ks_test.append( ks )
            #
        fig, ax = plt.subplots(figsize=(25, 10),nrows=1, ncols=1,sharey=True)
        #
        pearson = np.array(pearson)
        pearson = pearson[~np.isnan(pearson)]
        pearson = pearson.tolist()

        spearman = np.array(spearman)
        spearman = spearman[~np.isnan(spearman)]
        spearman = spearman.tolist()


        pearson_peo = np.array(pearson_peo)
        pearson_peo = pearson_peo[~np.isnan(pearson_peo)]
        pearson_peo = pearson_peo.tolist()

        square_error = np.array(square_error)
        square_error = square_error[~np.isnan(square_error)]
        square_error = square_error.tolist()

        square_error_peo = np.array(square_error_peo)
        square_error_peo = square_error_peo[~np.isnan(square_error_peo)]
        square_error_peo = square_error_peo.tolist()

        Ks_test = np.array(Ks_test)
        Ks_test = Ks_test[~np.isnan(Ks_test)]
        Ks_test = Ks_test.tolist()

        print(len(pearson))

        dum = []
        dum.append(pearson)
        dum.append(spearman)
        #dum.append(square_error)
        dum.append(pearson_peo)
        #dum.append(square_error_peo)
        dum.append(Ks_test)
        if self.lessdata==1:
            print("here")
            with open('lessdata.pkl', 'wb') as output:
                pickle.dump(dum, output, pickle.HIGHEST_PROTOCOL)
            print(len(lowdata))
        elif self.lessdata==0:
            with open('moredata.pkl', 'wb') as output:
                pickle.dump(dum, output, pickle.HIGHEST_PROTOCOL)
            print("here\n",len(moredata))
        dum = np.array(dum)
        dum = dum.T
        ax.boxplot(dum,patch_artist=True)
        ax.set_xticklabels(labels=['pearson','spearman','pearson_acc_Gene','Ks_test'], rotation=45, fontsize=10)
        ax.set_yticks([m/10 for m in range(-5,11,1)])
        plt.show()  
        #ax[1].scatter( [i for i in range(len(Ks_test))],Ks_test)
        #pdf.savefig(fig)
        #plt.close(fig)
        #pdf.close()
        return pearson, spearman, Ks_test, square_error
        #self.discriminate(pearson,inp_name,ter_name)
        #self.discriminate(pearson_peo,inp_name,ter_name+"ACC_GENE")   
    def gengraph(self,pred, target, inp_name, ter_name):
        pdf = mpdf.PdfPages(inp_name+" "+ter_name+'.pdf')
        fig, ax = plt.subplots(figsize=(30, 20),nrows=1, ncols=1,sharey=True)
        ax.set_xlabel('features')
        ax.set_ylabel("expression level")
        boxplot = []
        labels = []
        for i in range(len(pred[0])):
            boxplot.append([j[i] for j in pred])
            boxplot.append([j[i] for j in target])
        for i in range(len(pred[0])):
            labels.append(str(i+1))
            labels.append(str(i+1))
        boxplot = np.array(boxplot)
        boxplot= boxplot.T
        bx = ax.boxplot(boxplot,patch_artist=True,showfliers=False,vert=True)
        c = 0
        for box in bx['boxes']:
            c+=1
            if c%2==0 :
                box.set(color='red', linewidth=2)
                box.set(facecolor = 'green' )
                box.set(hatch = '/')
            else:
                box.set(color='green', linewidth=2)
                box.set(facecolor = 'red' )
                box.set(hatch = '/')

       
        ax.set_xticklabels(labels=labels,
                    rotation=90, fontsize=20)
        plt.legend([inp_name, ter_name], loc=0)
        pdf.savefig(fig)
        plt.close(fig)

        pearson_peo = []
        square_error_peo = []

        for i in range(len(pred)):
            p, _ = pearsonr(pred[i],target[i])
            pearson_peo.append(p)
            square_error_peo.append( (np.square(np.array(pred[i]) - np.array(target[i]))).mean(axis=0) )
        
        pearson = []
        spearman = []
        square_error = []
        Ks_test = []

        for k in range(len(target[0])):
            fig, ax = plt.subplots(figsize=(30, 20),nrows=1, ncols=2)
            fig.suptitle('features '+str(k+1))

            fpred = [i[k] for i in pred]
            frael = [i[k] for i in target]
            mn = min(min(fpred),min(frael))-5
            mx = max(max(fpred),max(frael))+5
            p, _ = pearsonr(fpred,frael)
            s,_ = spearmanr(fpred,frael)
            pearson.append( p )
            spearman.append( s)
            square_error.append( (np.square(np.array(fpred) - np.array(frael))).mean(axis=0) )

            x = np.linspace(mn, mx, num=100)

            r_density = gaussian_kde(frael)
            r_density.covariance_factor = lambda: .4
            #r_density._compute_covariance()

            Ltor_density = gaussian_kde(fpred)
            Ltor_density.covariance_factor = lambda: .4
            #Ltor_density._compute_covariance()

            ltor_cdf = Ltor_density(x)
            r_cdf = r_density(x)
            
            ks,_ = ks_2samp(fpred,frael)
            Ks_test.append( ks )

            ax[0].plot(x, ltor_cdf, color='blue')
            ax[0].plot(x, r_cdf, color='green')

            for i in range(1,len(ltor_cdf),1):
                ltor_cdf[i]+=ltor_cdf[i-1]

            
            for i in range(1,len(r_cdf),1):
                r_cdf[i]+=r_cdf[i-1]

            ax[0].set_ylabel('distribution')

            ax[1].plot(x, ltor_cdf, color='blue')
            ax[1].plot(x, r_cdf, color='green')

            ax[0].set_title('PDF')
            ax[1].set_title('CDF')


            plt.legend([ter_name, inp_name], loc=0)
            
            pdf.savefig(fig)
            plt.close(fig)

    
        fig, ax = plt.subplots(figsize=(25, 10),nrows=1, ncols=1,sharey=True)
        #plt.show()
        dum = []
        dum.append(pearson)
        dum.append(spearman)
        #dum.append(square_error)
        dum.append(pearson_peo)
        #dum.append(square_error_peo)
        dum.append(Ks_test)
        if self.lessdata==1:
            print("here")
            with open('lessdata.pkl', 'wb') as output:
                pickle.dump(dum, output, pickle.HIGHEST_PROTOCOL)
            print(len(lowdata))
        elif self.lessdata==0:
            with open('moredata.pkl', 'wb') as output:
                pickle.dump(dum, output, pickle.HIGHEST_PROTOCOL)
            print("here\n",len(moredata))
        dum = np.array(dum)
        dum = dum.T
        ax.boxplot(dum,patch_artist=True)
        ax.set_xticklabels(labels=['pearson','spearman','pearson_acc_Gene','Ks_test'],
                    rotation=45, fontsize=15)
                    
        #ax[1].scatter( [i for i in range(len(Ks_test))],Ks_test)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(25, 10), nrows=1, ncols=1, sharey=True)
        # plt.show()
        dum = []

        dum.append(square_error)
        dum.append(square_error_peo)

        dum = np.array(dum)
        dum = dum.T
        ax.boxplot(dum, patch_artist=True)
        ax.set_xticklabels(labels=[ 'MSE', 'MSE_acc_Gene'],
                           rotation=45, fontsize=15)

        ax[1].scatter( [i for i in range(len(Ks_test))],Ks_test)
        pdf.savefig(fig)
        plt.close(fig)
        pdf.close()
        #self.discriminate(pearson,inp_name,ter_name)
        return pearson, spearman, Ks_test, square_error

        
        

    def test_LtoR(self):
       
        r_c0 = [[0 for i in range(len( self.l_c[0]) )] for j in range(len(self.l_c))]
        l_tc = np.array(self.l_c)
        r_c0 = np.array(r_c0, dtype =np.float)
        a,b,c,d,e,f,h = self.model.predict([l_tc,r_c0 ])
        #y = d.tolist()
        #self.gengraph(f,self.r_c ,'Right PCA vs','Predicted_right_PCA')
        f = self.transformation.inv_trans_right(f)# use if i reduce the dimention of data
        return self.gengraph_bigdata(f,self.real_rc ,'Right vs','Predicted_right')
        #self.gengraph_bigdata(self.transformation.inv_trans_right(self.r_c),self.real_rc ,'Right','vs REAL_PCA_right')

        
    def test_LtoL(self):
        l_c0 = [[0 for i in range(len(self.r_c[0]))] for j in range(len(self.r_c))]
        l_tc = np.array(self.l_c)
        l_c0 = np.array(l_c0, dtype=np.float)
        a, b, c, d, e, f, h = self.model.predict([l_tc, l_c0])
        #self.gengraph(a, self.l_c, 'Left_PCA', 'vs predicted_left_PCA-from left')
        a = self.transformation.inv_trans_left(a)
        return self.gengraph_bigdata(a, self.real_lc, 'Left', 'vs predicted_left-from left')
        #self.gengraph_bigdata(self.transformation.inv_trans_left(self.l_c), self.real_lc, 'Left', 'vs REAL_pca_left')

    def test_RtoL(self):
        l_c0 = [[0 for i in range(len( self.r_c[0]) )] for j in range(len(self.r_c))]
        r_tc = np.array(self.r_c)
        l_c0 = np.array(l_c0, dtype =np.float)
        a,b,c,d,e,f,h = self.model.predict([l_c0,r_tc ])
        #self.gengraph(b, self.l_c, 'Left pca vs ', 'predicted_left_pca')
        b = self.transformation.inv_trans_left(b)
        self.gengraph_bigdata(b,self.real_lc ,'Left','vs predicted_left')
        #self.gengraph_bigdata(self.transformation.inv_trans_left(self.l_c),self.real_lc ,'Left','From_pca_left')

    def test_U_RtoL(self):
        l_u0 = [[0 for i in range(len( self.r_u[0]) )] for j in range(len(self.r_u))]
        r_tc = np.array(self.r_u)
        l_c0 = np.array(l_u0)
        a,b,c,d,e,f,h = self.model.predict([l_c0,r_tc ])
        self.gengraph(c,self.l_u_ ,'U_right','U_rightTOleft')
    
    def test_U_LtoR(self):
       
        r_c0 = [[0 for i in range(len( self.l_u[0]) )] for j in range(len(self.l_u))]
        l_tc = np.array(self.l_u)
        r_c0 = np.array(r_c0)
        a,b,c,d,e,f,h = self.model.predict([l_tc,r_c0 ])
        #y = d.tolist()
        self.gengraph(f,self.r_u_ ,'U_left','U_right')
def copula_cmp():
    with open('lessdata.pkl', 'rb') as input:
        lowdata = pickle.load(input)
    with open('moredata.pkl', 'rb') as input:
        moredata = pickle.load(input)
    All = []
    print(len(lowdata), len(moredata))
    for i in range(len(lowdata)):
        All.append(lowdata[i])
        All.append(moredata[i])
    All = np.array(All)
    fig, ax = plt.subplots(figsize=(30, 20),nrows=1, ncols=1,sharey=True)
    All = All.T
    ax.boxplot(All,patch_artist=True)
    ax.set_xticklabels(labels=['pearson_l','pearson_m','spearman_l','spearman_m','MSE_l','MSE_m',
    'pearson_acc_people_l','pearson_acc_people_m','MSE_acc_people_l','MSE_acc_people_m','Ks_test_l','Ks_test_m'],
                rotation=45, fontsize=15)
    fig.savefig("copulatest.jpg")
        

def boxplot(values, names, title):
    '''values = [
        [1,2,3,4,3,1,3,4,3,1,2,3,4,23,1,2,3],
        [10, 20, 30, 40, 30, 10, 30, 40, 30, 1, 2, 37, 47, 23, 1, 2, 3],
        [-1, -2, -4, -6, 0 , 1, 3 , 6 , 7, -9,10,11]
    ]'''
    values = np.array(values)
    #name = ["a", "b", "c"]
    plt.figure(figsize=(60, 20))
    import seaborn as sns
    ax = sns.boxplot(data = values, orient='v')
    ax.set_xticklabels(labels=names,
                    rotation=60, fontsize=18)
    ax.get_figure().savefig(title+".png")



        

