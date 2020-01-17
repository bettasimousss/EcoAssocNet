# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:27:33 2019

@author: simoussi
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from .Util import poly, get_scaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

try:
    from skmultilearn.model_selection import iterative_train_test_split
except:
    print("Scikit-multilearn not installed !")


class DataPrep(object):
    def __init__(self,num_std="standard",cat_trt="fca"):
        self.num_std=num_std
        self.cat_trt=cat_trt
        self.cat_encoder=[]  ##to use on test set
        self.num_stdizer=[] ##to use on test set
        self.fca=None
        self.kfolds=[]
        self.groups={}
        self.groups_id=pd.Series(dtype=int)
        
    def load_dataset(self,feat,occur,num,cat):
        self.features=feat
        self.targets=occur
        self.num_atts=num
        self.cat_atts=cat
        self.n, self.m = occur.shape
        self.num_features=self.features.loc[:,num]
        self.cat_features=self.features.loc[:,cat] 
        self.names=occur.columns.tolist()
        
    def create_poly(self,deg,featlist):
        polyfeats=[]
        polynames=[]
        for f in self.num_atts:
            if(f in featlist):
               poly_f=poly(self.features[f].values.reshape(self.n,1),deg)
               polyfeats.append(poly_f)
               polynames.extend([f+"_"+str(i+1) for i in range(deg)])
            else:
               polyfeats.append(self.features[f].values.reshape(self.n,1))
               polynames.append(f)
        
        self.num_features=pd.DataFrame(data=np.concatenate(polyfeats,axis=1),columns=polynames)
        self.num_atts=polynames
        
    def preprocess_numeric(self):
        cpt=len(self.groups_id)
        for f in range(len(self.num_atts)):
            if(self.num_std[f]!=None):
                scaler=get_scaler(self.num_std[f])
                self.num_features[self.num_atts[f]]=scaler.fit_transform(self.num_features[self.num_atts[f]].values.reshape(self.n,1))
                self.num_stdizer.append(scaler)
            else:
                self.num_stdizer.append("")
            
            self.groups[self.num_atts[f]]=[self.num_atts[f]]
            self.groups_id[self.num_atts[f]]=cpt
            cpt+=1
                    
    
    def process_categoric(self,fcacomp=2): 
        cat_encodings=[]
        for f in self.cat_atts:
            encoder=LabelEncoder()
            cat_encodings.append(encoder.fit_transform(self.features[f].values))
            self.cat_encoder.append(encoder)
        
        self.cat_features=pd.DataFrame(data=np.stack(cat_encodings,axis=1),columns=self.cat_atts)
        
        if (self.cat_trt=="fca"):
            fca=FactorAnalysis(n_components=fcacomp)
            self.cat_atts=["fca_"+str(i) for i in range(fcacomp)]
            self.cat_features=pd.DataFrame(data=fca.fit_transform(self.cat_features.values),columns=self.cat_atts)
            self.fca=fca
            
            cpt=len(self.groups_id)
            for f in range(fcacomp):
                self.groups[self.cat_atts[f]]=[self.cat_atts[f]]
                self.groups_id[self.cat_atts[f]]=cpt
                cpt+=1
            
        elif(self.cat_trt=="onehot"):
            onehotenc=OneHotEncoder(categories='auto')
            onehotenc.fit(self.cat_features)
            self.onehotenc=onehotenc
            data=onehotenc.transform(self.cat_features).todense()
            cat_feats=[]
            for f in range(len(onehotenc.categories_)):
                cat_feats.extend([self.cat_atts[f]+"_"+str(int(i)) for i in onehotenc.categories_[f]])
            
            cpt=len(self.groups_id)
            for catft in self.cat_atts:
                ### Group dummies ###
                dummies=[x for x in cat_feats if x.find(catft)==0] 
                self.groups[catft]=dummies
                self.groups_id=self.groups_id.append(pd.Series([cpt]*len(dummies),index=dummies,dtype=int))
                cpt+=1
                                                                                                                                         
            self.cat_atts=cat_feats
            self.cat_features=pd.DataFrame(data=data,columns=self.cat_atts)
        
            
    def combine_covariates(self):
        self.covariates=pd.concat([self.num_features,self.cat_features],axis=1)
        self.p=self.covariates.shape[1]
        
    def train_test_split(self,meth="random",prob=0.8,nf=5):
        if(meth=="stratified"):  ##Use scikit-multilearn
            train, _, test, _ = iterative_train_test_split(np.arange(0,self.n).reshape(self.n,1), self.targets.values, test_size = 1-prob)
            self.idx_train=train[:,0]
            self.idx_test=test[:,0]
        elif(meth=="random"):
            s=np.random.choice(a=2,size=self.n,p=[prob,1-prob])
            self.idx_train=np.where(s==0)[0]
            self.idx_test=np.where(s==1)[0]
        elif(meth=="kfold"):
            indices=np.arange(self.n)
            np.random.shuffle(indices)
            foldsize=int(np.round(self.n/nf))
            for i in range(nf):
                self.kfolds.append({'fold':i+1,
                                    'idx_test':indices[np.arange(start=i*foldsize,stop=(i+1)*foldsize).tolist()],
                                    'idx_train':indices[np.arange(start=0,stop=i*foldsize).tolist()+np.arange(start=(i+1)*foldsize,stop=self.n).tolist()]
                        })
        elif(meth=="bootstrap"):
            indices=np.arange(self.n)
        else: ###do not split
            self.idx_train=np.arange(0,self.n)
            self.idx_test=np.arange(0,self.n)
            
    def pretrain_glms(self,cv=1,penalty='l1',class_weight='balanced',max_iter=300,fit_intercept=True,solver='liblinear'):
        params=[]
        perfs=[]
        
        ### create indices of train and test ###
        indices=np.arange(self.n)
        np.random.shuffle(indices)
        foldsize=int(np.floor(self.n/cv))
        
        ### Train and test on each train-test fold ###
        for i in range(cv):
            if(cv==1):
                idx_train=idx_test=indices
            else:
                idx_train=indices[np.arange(start=0,stop=i*foldsize).tolist()+np.arange(start=(i+1)*foldsize,stop=self.n).tolist()]
                idx_test=indices[np.arange(start=i*foldsize,stop=(i+1)*foldsize).tolist()]
            
            ### Train ###
            weights=[]
            auc_list=[]
            biases=[]
            m=self.m
            X_train=self.covariates.loc[idx_train].values
            X_test=self.covariates.loc[idx_test].values
            Y_train=self.targets.loc[idx_train]
            Y_test=self.targets.loc[idx_test]
            
            for s in range(m): 
                Ys=Y_train.loc[idx_train,self.names[s]].values
                lr=LogisticRegression(penalty=penalty,class_weight=class_weight,max_iter=max_iter,fit_intercept=fit_intercept,solver=solver)
                lr.fit(X_train,Ys)
                w=lr.coef_[0]
                inter=lr.intercept_
                weights.append(w)
                biases.append(inter)
                Y_pred=lr.predict(X_test)
                try:
                    auc=roc_auc_score(Y_test.loc[idx_test,self.names[s]],Y_pred)
                except ValueError:
                    auc=-1                  
                
                auc_list.append(dict(species=self.names[s],score=auc))
            
            aucs=pd.DataFrame.from_dict(auc_list)
            weights_hsm=np.stack(weights)
            
            perfs.append(aucs)
            params.append({'w':weights_hsm,'b':biases})
        
        return perfs, params           
        
