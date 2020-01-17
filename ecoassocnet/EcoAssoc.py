# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:22:32 2019

@author: saras
"""

import sys
sys.path.append("Model")
from Model.init_config import init_model_from_paramfile
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from Model.zie import fit_emb, evaluate_emb
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from keras.callbacks import TensorBoard
from Util.Util import sigmoid, compute_deviance, compute_AIC, plot_archi_hsm#, concat_weights
from scipy import sparse
import configparser

np.random.seed(135545) ##For reproducibility

'''
Defaut configuration of hyperparameters
'''

def parse(x):
    if x=='False':
        return False
    
    if x=='True':
        return True
    try:
        if '.' not in x:
            return int(x)
        else:
            return float(x)
    
    except:
        return x
    
def load_default_config(file='hsm_archi.ini'):
    confpars = configparser.ConfigParser()
    
    confpars.read(file)
    
    config={}
    for s in confpars.sections():
        config.update({x[0]:parse(x[1]) for x in confpars.items(s)})

    return config

config_def=load_default_config('default.ini')

#config_def= dict(
#          K=2,    ### Embedding dimension
#          
#          #HSM model
#
#          exposure=True,  ### Whether to fit a habitat suitability model
#          use_covariates=True, ### Set to False if homogeneous environment, will fit only the intercept
#          intercept=False, ## Set to true if an intercept is to be fitted for the HSM component 
#          fixedoccur=False, ### Set to true if no hsm is fit but instead habitat suitability scores are given. Is ignored if exposure=False
#          w_sigma2=1, ### Variance of weights 
#          
#          # Abundance model
#          bias=True, ### Set to true if a bias is used in the case of empty biotic contexts
#          offset=False, ### Set to true if this bias if fixed
#          dist='poisson', ### Probability distribution of used data
#          
#          #Association plasticity
#          assoc_plasticity=False, ### Set to true if a betassoc model needs to be fit
#          
#          #Embeddings regularization
#          use_reg=True, ###Set to true if we want regularization to be applied to the learnt embeddings
#          ar_sigma2=1, ### Variance of embeddings 
#          prior="gaussian",
#          lambda_lasso=0.1,
#          use_penalty=False, ##Unused for now
#          emb_initializer="uniform", 
#          fixed_rho=False, ###Set to true if the response embeddings are not fitted Associations are determined by source effects only.
#          
#          #Training parameters
#          optim="sgd",  ###optimizer used amongst supported Tensorflow optimizers See: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
#          sample_ratio=1, 
#          use_valid=False, ###Set to true if during training, a subsample of the data is to be kept apart for validation
#          
#          LR=0.01, ##Learning rate
#          use_decay=False,   ##Whether to use learning rate decay as a function of the validation loss
#          lr_update_step=10000, ##Frequency of updates
#          lr_update_scale=0.5, ##Scale of updates
#          
#          #Training duration
#          batch_size=1, ##Batch size, Online stochastic gradient descent used batches with single observation (single community)
#          max_iter=200000, ##Maximum number of iterations = data_size/batch_size * number of epochs (full evaluation of the training set)
#          nprint=10000) ###Number of batches prediction until a score is printed (if verbose=1), also until a validation is done

class EcoAssoc(object):
    def __init__(self,config='default.ini',labels=[],name_dataset="dataset",target="count"):
        self.hsm_model=None  ##HSM model parameters
        self.mle=None ##Best model so far
        self.config=load_default_config(config) ##Configuration
        self.trained_hsm=False ##Whether the used HSM is trained
        self.trained_im=False ##Whether the association model is trained 
        self.species_names=labels
        self.m=len(labels)
        self.name_dataset=name_dataset
        self.target=target
    
    def set_mle(self,mle):
        self.mle=mle
    
    def set_hsm(self):
        _types, self.hsm_model, _=init_model_from_paramfile(self.config['archi_desc_file'], self.config['archi_plot_file'],embed_size=self.config['k'],plot=False,plasticity=self.config['assoc_plasticity'])
        if self.trained_im:
            self.hsm_model.set_weights(self.mle['hsm'])
    
    def pretrain_hsm(self,hsm_config_file,train_params,dataset,tb="",verbose=1,outfile="hsm_model.png",plot_archi=False):
        ## If HSM is specified as a keras model
        ## Use this helper function to pretrain it
        
        _types, self.hsm_model, _=init_model_from_paramfile(hsm_config_file, outfile,embed_size=self.config['k'],plot=plot_archi,plasticity=self.config['plasticity'])
        X_train=dataset['X_train']
        Y_train=dataset['Y_train']
        X_test=dataset['X_test']
        Y_test=dataset['Y_test']
        
        w=train_params['w']
        met=train_params['met']
        optim=train_params['opt']
        l=train_params['l']
        bs=train_params['bs']
        th=train_params['th']
        ep=train_params['ep']

        if(w):
            prevs=Y_train.sum(axis=0)/Y_train.shape[0]
            lw=[(1-prevs[i])/prevs[i] for i in range(self.m)]
        else:
            lw=None 
        Y_tr=[Y_train.iloc[:,i].values.reshape(-1,1) for i in range(self.m)]
        
        cbk=[]
        if(tb!=""):
            tensbd=TensorBoard(log_dir=tb,batch_size=bs)
            cbk.append(tensbd)
            
        self.hsm_model.compile(optimizer=optim,loss=l,metrics=met,loss_weights=lw)
        self.hsm_model.fit(X_train,Y_tr,batch_size=bs,epochs=ep,verbose=verbose,callbacks=cbk)
        predictions=self.hsm_model.predict(x=X_test)  
        auc_scores=[]
        for i in range(len(predictions)):
            score=roc_auc_score(Y_test.iloc[:,i],predictions[i])
            auc_scores.append(score)
            
        preds=np.concatenate(predictions,axis=1)
        Y_pred=sigmoid(preds)
        
        try:
            macroauc=roc_auc_score(Y_test,Y_pred,"macro")
        except ValueError:
            macroauc=-1
        
        microauc=roc_auc_score(Y_test,Y_pred,"micro")
            
        accuracies=[accuracy_score(Y_test.values[:,i],(Y_pred>th).astype(np.int32)[:,i]) for i in range(self.m)]
        
        return dict(macroauc=macroauc, microauc=microauc, accuracy=np.mean(accuracies), aucs=auc_scores)
        
    def plot_hsm(self,summary=False,image=False):
        if summary: self.hsm.summary()
        if image: plot_archi_hsm(self.hsm)
        
    
    def update_config(self,key,value):
        self.config.updat({key:value})
        
    def train_interaction_model(self,dataset=(),verbose=1,init_weights=None,offset=None):
        trainset=dict(scores=sparse.csr_matrix(dataset[1].astype(np.float32)),
                          atts=[dataset[0].astype(np.float32)])
        
        if(self.trained_hsm):
            weights=self.hsm_model.get_weights()
        else:
            weights=[]
            p=init_weights.shape[1]-1
            for i in range(init_weights.shape[0]):
                weights.append(init_weights[i,1:].reshape(p,1))
                weights.append(init_weights[i,0].reshape(1,))
        
        emb_model, logg, mle, sinds, flags= fit_emb(reviews=trainset, config=self.config, verbose=verbose,init_weights=weights,offset=offset)
        
        self.mle=mle
        self.trained_im=True
        self.trained_hsm=True
        
        self.set_hsm()
        return logg, sinds, flags
      
    def evaluate_model(self,testdata=()):
        testset=dict(scores=sparse.csr_matrix(testdata[1].astype(np.float32)),
                         atts=[testdata[0].astype(np.float32)])
        testperfs = evaluate_emb(testset,self.mle,self.config)
        C_test=testset['scores'].todense()
        Y_test=(C_test>0).astype(int)
        
        ### Compute performance metric ###
        posllh=[np.mean(testperfs[1][i]) for i in range(len(C_test))]
        occur_probs=sigmoid(testperfs[2])
        score=np.mean(testperfs[0])
        
        logmeanpos=testperfs[3]
        posidx=testperfs[4]
        
        
        ### Performances of HSM component ###
        perf_hsm=dict(microauc=roc_auc_score(Y_test,occur_probs,"micro"))
        
        prevs=(Y_test.sum(axis=0)/Y_test.shape[0]).tolist()[0]
        
        if(self.target=="count"):
            dev=[np.mean(compute_deviance(y=C_test[i,posidx[i][0].tolist()].astype(int),u=logmeanpos[i])) for i in range(len(posidx))]
            pos_deviance=np.nanmean(dev)
        
            perf_im=dict(
                avgposllh=np.nanmean(posllh),
                avgllh=score,
                pos_deviance=pos_deviance, ##averaged over examples of deviance
                deviances=dev, ##deviance when poisson or negbin are used
                AIC=compute_AIC(logllh=score,pool_size=C_test.shape[1],embed_size=self.config['k'],p=3,intercept_fit=False)
                )
        else:
            probs=np.zeros(Y_test.shape)
            for i in range(len(Y_test)):
                logmeanneg=testperfs[6][i][1]
                negidx=testperfs[6][i][0][:,1]
                probs[i,negidx]=sigmoid(logmeanneg)
                probs[i,posidx[i][0]]=np.exp(logmeanpos[i])
        
            Y_pred=occur_probs*probs
            microauc_all=roc_auc_score(Y_test,Y_pred,"micro")
        
            accs=[accuracy_score(Y_test[:,j],Y_pred[:,j]>0.5) for j in range(Y_pred.shape[1])]
            accuracy=np.mean([x*y for x,y in zip(accs,prevs)])
        
            perf_im=dict(avgposllh=np.nanmean(posllh),
                         avgllh=score,
                         microauc_all=microauc_all,
                         accuracies=accs,
                         waccuracy=accuracy)
            
        return perf_hsm, perf_im
        
#        microauc=roc_auc_score(Y_test,occur_probs,"micro")
#        microauc_all=roc_auc_score(Y_test,occur_probs*(Y_pred>=1),"micro")
#        
#        macroauc=np.nanmean([accuracy_score(Y_test[:,j],occur_probs[:,j]) for j in range(Y_pred.shape[1])])
#        macroauc_all=np.nanmean([accuracy_score(Y_test[:,j],((Y_pred>=1) * occur_probs)[:,j]) for j in range(Y_pred.shape[1])])
#        
#        try:
#            macroauc=roc_auc_score(Y_test,occur_probs,"weighted")
#        except ValueError:
#            macroauc=-1
            
#        perf_im=dict(
#            avgposllh=np.nanmean(posllh),
#            microauc=(microauc,microauc_all),
#            macroauc=(macroauc,macroauc_all),
#            avgllh=score,
#            pos_deviance=pos_deviance, ##averaged over examples of deviance
#            deviances=dev, ##deviance when poisson or negbin are used, accuracy otherwise
#            AIC=compute_AIC(logllh=score,pool_size=C_test.shape[1],embed_size=self.config['k'],p=3,intercept_fit=False)
#        )

    def compute_associations(self,norm=False,save=True,file=None):
        
        if(norm):
            assoc=cosine_similarity(self.mle['rho'],self.mle['alpha'])
        else:
            assoc=np.dot(self.mle['rho'],self.mle['alpha'].T)
        
        assoc_df=pd.DataFrame(assoc,columns=self.species_names,index=self.species_names)
        if(save):
            if(file==None): file="association_matrix"
            assoc_df.to_csv(file,sep=";",decimal=".",index=True)
            
        return assoc_df
        
    
    def compute_associations_onX(self,X):
        betas=self.mle['betassoc'][0]
        
        assoc_weights=np.dot(X,betas) ###n,K  ==> n,1,K
        alphas=self.mle['alpha']  ###m,K ==>1,m,K        
        
        rhos=self.mle['rho']   ###m,K  
        context_effects=np.expand_dims(assoc_weights,axis=1)*np.expand_dims(alphas,axis=0)  ###n,m,K
        
        assocs=np.dot(context_effects,rhos.T) ### n,m,m
        
        return assocs
          
    
    def save_embeddings(self,file_rho="rho",file_alpha="alpha"):
        pd.DataFrame(self.mle['rho']).to_csv(file_rho,sep=";",decimal=".")
        pd.DataFrame(self.mle['alpha']).to_csv(file_alpha,sep=";",decimal=".")
        
    def predict(self,X,C,Yc,t):
        hsscore=sigmoid(self.hsm_model.predict(X))[:,t]
        rho=self.mle['rho'][t,:]
        ctx=np.dot(Yc,self.mle['alpha'][C,:])/len(C)
        emb=np.dot(ctx,rho)+self.mle['invmu'][t]
        if(self.config['dist'] in ["poisson","negbin"]):
            mean=np.maximum(emb,0)
        elif(self.config['dist'] in ["binomial"]):
            mean=sigmoid(emb)
        
        return hsscore, mean    
