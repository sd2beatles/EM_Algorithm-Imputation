import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix 
from itertools import product
import sys
import pandas as pd



class EMImpute(object):
    def __init__(self,data,max_iter=10,k=3):
        self.nData=data.shape[0]
        self.nDim=data.shape[1]
        self.data=data
        self.k=k
        self.max_iter=max_iter

        
        
    def initialize(self):
        #find the rows and columns of missing values
        m=np.isnan(self.data)==True
        self.missing_row=np.where(m==True)[0]
        self.missing_col=np.where(m==True)[1]
        
        #initialize mean and covariance for nDim times
        col_samples=np.random.choice(self.nDim,self.nDim)
        self.mu=[np.nanmean(self.data,axis=0) for _ in range(self.k)]
        self.sigma=[np.asarray(make_spd_matrix(n_dim=self.nDim,random_state=42)) for _ in range(self.k)]
    

    def e_step(self):
        likelihoods=np.zeros((self.nData,self.k))
        for c in range(self.k):
            likelihood=list()
            for i in range(self.nData):
                observed_cols=self.observed(i)
                z=self.data[i,observed_cols]
                pdf=self.multivariate_normal(z,observed_cols,c)
                likelihood.append(pdf)
            likelihoods[:,c]=likelihood
        weights=likelihoods/likelihoods.sum(axis=1).reshape(-1,1)
        self.weights=weights
    
  
    def m_step(self):
        for j in range(self.k):
            weight=self.weights[:,j]
            weight_sum=weight.sum()
            C=np.zeros((self.nDim,self.nDim))
            for i in range(self.nData):
                sigma_MM,sigma_OO,sigma_MO,missing_cols=self.imputation(i,j)
                part1=self.weights[i,j]/weight_sum
                part2=sigma_MM-np.dot(np.dot(sigma_MO,np.linalg.inv(sigma_OO)),sigma_MO.T)
                try:
                    C_updated=C[np.ix_(missing_cols,missing_cols)]+part1*part2
                    C_updated=np.hstack(C_updated)
                except:
                    pass
                #compute the combinations of missing columns
                all_cols=[missing_cols,missing_cols]
                comb=product(*all_cols)
                for k,l in zip(comb,C_updated):
                    C[k]+=l
                    
            self.mu[j]=(np.dot(self.data.T,weight)/weight_sum)
            data_mu=(self.data-self.mu[j])
            temp=weight.reshape(-1,1)*data_mu
            self.sigma[j]=np.dot(temp.T,temp)+C
            
    
    def fit_transform(self):
        self.initialize()
        for iteration in range(self.max_iter):
            self.e_step()
            self.m_step()
        return self.data
                
    def multivariate_normal(self,z,observed_cols,k_no):
  
        sigma=self.sigma[k_no][np.ix_(observed_cols,observed_cols)]
        L=np.linalg.cholesky(sigma)
        L_inv=np.linalg.inv(L)
        # make a use of cholesky decomposition to compute sigma inverse of covariance of observed 
        #matrix. This helps to increase an accuracy and stability of the computation
        const=1/np.sqrt(2*np.pi*np.linalg.norm(np.dot(L_inv.T,L_inv)**self.nDim))
        z=z-self.mu[k_no][observed_cols]
        const2=np.exp(np.linalg.norm(np.dot(L_inv,z),2))
        likelihood=const*const2
    
            
        return likelihood

        
    def observed(self,row):
        if row in self.missing_row:
            missing_rows=np.where(self.missing_row==row)
            missing_cols=self.missing_col[missing_rows]
            observed_cols=np.delete(np.arange(self.nDim),missing_cols)
        else:
            observed_cols=np.arange(self.nDim)
        return observed_cols
              
                
    
    def imputation(self,row,k_no):
        #find the mu and sigma for a given class
        mu=self.mu[k_no]
        sigma=self.sigma[k_no]
        
        #specify the observed and missing columns for a given row
        observed_cols=self.observed(row)
        missing_index=np.where(row==self.missing_row)
        missing_cols=self.missing_col[missing_index]
        
        #compute the relevant components
        sigma_MM=sigma[np.ix_(missing_cols,missing_cols)]
        sigma_OO=sigma[np.ix_(observed_cols,observed_cols)]
        sigma_MO=sigma[np.ix_(missing_cols,observed_cols)]
        observed_data=self.data[row,observed_cols]
        observed_mu=mu[observed_cols]
        part1=np.dot(sigma_MO,np.linalg.inv(sigma_OO))
        part2=self.data[row,observed_cols]-mu[observed_cols]
        self.data[row,missing_cols]=np.dot(part1,part2)
        return sigma_MM,sigma_OO,sigma_MO,missing_cols
    
    
        
