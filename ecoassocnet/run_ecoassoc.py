import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from .Util.DataPrep import DataPrep
from .EcoAssoc import EcoAssoc
from .Util.Util import compute_offset
from .Util.Util import cooccur, response_sim, biogeo_filter, plot_dendrograms

def load_data(folder_data,file_env,file_count,num_vars,cat_vars,num_std="minmax",cat_trt="onehot"):
    env=pd.read_csv(folder_data+file_env,sep=";",decimal=".")
    counts=pd.read_csv(folder_data+file_count,sep=";",decimal=".")
    names=counts.columns.tolist()
    occur=(counts>0).astype(int)
    counts.head()
    
    prep=DataPrep(num_std=[num_std]*len(num_vars),cat_trt=cat_trt)
    prep.load_dataset(feat=env,occur=occur,num=num_vars,cat=cat_vars)
    prep.preprocess_numeric()
    prep.process_categoric()
    prep.combine_covariates()
    
    return prep, names, counts, occur


def pretrain_hsm(prep):
    perfs,params=prep.pretrain_glms()
    biases=np.array(params[0]['b'])
    weights=np.concatenate([biases,params[0]['w']],axis=1)
    weights_df=pd.DataFrame(data=weights,columns=['bias']+prep.covariates.columns.tolist())
    
    return weights_df

def training_data_split(prep,counts,meth="stratified",prob=0.8):
    prep.train_test_split(meth=meth,prob=prob)
    X_train=prep.covariates.iloc[prep.idx_train,:].values
    X_test=prep.covariates.iloc[prep.idx_test,:].values
    
    Y_train=counts.iloc[prep.idx_train,:].values
    Y_test=counts.iloc[prep.idx_test,:].values
    
    return dict(train=(X_train,Y_train),test=(X_test,Y_test))


def run_ecoassoc(folder_data,file_env,file_count,training_config_file,
                 name_dataset="dataset",target="count",
                 num_vars=None,cat_vars=None,num_std="minmax",cat_trt="onehot",
                 offset_mode=None,verbose=1,
                 meth="stratified",p=0.8):
    
    print("Preparing data")
    prep, names, counts, occur=load_data(folder_data,file_env,file_count,num_vars,cat_vars,num_std,cat_trt)
    init_weights=pretrain_hsm(prep)
    
    dataset=training_data_split(prep,counts,meth=meth,prob=p)
    offsets=compute_offset(counts,offset_mode)
    
    ### Train ###
    print("Training")
    ecoasso_model=EcoAssoc(config=training_config_file,labels=names,name_dataset=name_dataset,target=target)
    logg= ecoasso_model.train_interaction_model(dataset=dataset['train'],verbose=verbose,init_weights=init_weights.values,offset=offsets)
    perf_hsm, perf_im=ecoasso_model.evaluate_model(testdata=dataset['test'])
    
    final_weights=ecoasso_model.mle['hsm']
    
    weights=np.concatenate([final_weights[i] for i in range(len(final_weights)) if i%2==0],axis=1)
    weights_df=pd.DataFrame(weights.T,columns=prep.covariates.columns.tolist(),index=names)
    
    ecoasso_model.mle['final_weights']=weights_df
    
    return dict(metadata=dict(names=names),data=dict(env=prep.covariates,count=counts,occur=occur),model=ecoasso_model, train_log=logg, test_perf=(perf_hsm,perf_im))


def apply_biogeo_filter(assoc_df,occur,weights,thoccur,thass,thresp,names):
    cooc=cooccur(occur,names)
    respsim=response_sim(weights)
    selected_assoc=biogeo_filter(assoc_df,cooc,respsim,thoccur,thass,thresp,len(names))
    
    return selected_assoc