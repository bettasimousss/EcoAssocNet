# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:33:30 2019

@author: Sara Si-moussi - sara.si-moussi@inra.fr sara.simoussi@gmail.com

Edited: October 2019
"""

import tensorflow as tf
import numpy as np
tf.set_random_seed(1234)
np.random.seed(100)

# Multiple Outputs
from keras.models import Model
from keras.regularizers import l1, l2, l1_l2
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Embedding, Concatenate, Activation, Dropout

####################################### DL Model ##############################################

def get_regularizer(regtype,regparams):
    if(regtype=="l1"):
        kr=l1(l=regparams[0])
    elif(regtype=="l2"):
        kr=l2(l=regparams[0])
    elif(regtype=="l1_l2"):
        kr=l1_l2(l1=regparams[0],l2=regparams[1])
    else:
        kr=None 
    return kr

### Feature extraction components 
def spatial_fe_comp(spat_params,feat_name):  ##CNN alternating convolution and pooling layers
    imdimensions=spat_params["imsize"]
    imchannels=spat_params["nbchannels"]
    
    nb_alt=spat_params["nbalt"]  ##Number of convolution, pooling alternations

    conv_params=spat_params["conv"]
    pool_params=spat_params["pool"]
    
    activs=spat_params["activ"]
    
    fc_params=spat_params["fc"]
    
    reg=spat_params["reg"]
    
    krcnn=get_regularizer(reg.get("regtype")[0],reg.get("regparam")[0])
    krfc=get_regularizer(reg.get("regtype")[1],reg.get("regparam")[1])

    ### 1.Input
    input_raster=Input(shape=(imdimensions,imdimensions,imchannels),name="in_"+feat_name,dtype=tf.float32)
    
    ### 2.Convolutions + Pooling
    prevalt=Dropout(rate=reg.get("dropout")[0])(input_raster)
    for i in range(nb_alt):
        prevalt=Conv2D(conv_params.get("nbfilt")[i], kernel_size=conv_params.get("fsize")[i], strides=conv_params.get("cs")[i], padding=conv_params.get("cp")[i],kernel_regularizer=krcnn,name=feat_name+"_conv_"+str(i+1))(prevalt)
        prevalt=MaxPool2D(pool_size=pool_params.get("psize")[i], strides=pool_params.get("ps")[i], name=feat_name+"_pool_"+str(i+1))(prevalt)
        prevalt=Activation(activs[i])(prevalt)
        prevalt=Dropout(rate=reg.get("dropout")[1])(prevalt)

    ### 3. Flattening layer
    prevalt=Flatten()(prevalt)
    
    ### 4. Fully connected layer
    for i in range(fc_params.get("nbfc")):
        prevalt=Dense(fc_params.get("nnfc")[i], activation=fc_params.get("actfc"),kernel_regularizer=krfc,name=feat_name+"_fc_"+str(i+1))(prevalt)
        prevalt=Dropout(rate=reg.get("dropout")[2])(prevalt)
    
    cnn_model=Model(input_raster,prevalt)
    
    return cnn_model

def cat_emb_comp(cat_param,feat_names): ##Embedding layers for categorical features and text
    reg=get_regularizer(cat_param.get("reg").get("regtype"),cat_param.get("reg").get("regparam"))
    l_in=[]
    l_out=[]
    for i in range(cat_param.get("nbc")):
        in_cat=Input(shape=(1,),name=feat_names[i],dtype=tf.int32)
        embed_cat=Embedding(input_dim=cat_param.get("nmod")[i],input_length=1,output_dim=cat_param.get("embed_size")[i],name="embed_"+feat_names[i],embeddings_regularizer=reg)(in_cat)
        embed_cat_resh=Flatten()(embed_cat)
        l_in.append(in_cat)
        l_out.append(embed_cat_resh)
    
    out=Concatenate()(l_out)
        
    return(Model(l_in,out))

def num_fc_comp(num_param,nm):
        
    in_num=Input(shape=(num_param.get("nbnum"),),name=nm,dtype=tf.float32)
    reg=get_regularizer(num_param.get("reg").get("regtype"),num_param.get("reg").get("regparam"))
    
    prev=in_num
    for i in range(num_param.get("nl")):
        prev=Dense(num_param.get("nn")[i], activation=num_param.get("num_act"),name=nm+"_"+str(i),kernel_regularizer=reg)(prev)
        prev=Dropout(num_param.get("reg").get("dropout"))(prev)
    
    m=Model(in_num,prev)
    
    return(m)
              
def feat_transf_comp(ft_params,fe_comps): ##Transforming extracted features (shared component)
    fe_ins=[]
    fe_out=[]
    for c in fe_comps:
        if(type(c.input)==list):
            fe_ins.extend(c.input)
        else:
            fe_ins.append(c.input)
        fe_out.append(c.output)
    
    if(len(fe_out)>1):
        prev=Concatenate()(fe_out)
    else:
        prev=fe_out[0]
    reg=get_regularizer(ft_params.get("reg").get("regtype"),ft_params.get("reg").get("regparam"))
    for i in range(ft_params.get("nl")):
        prev=Dense(ft_params.get("nn")[i], activation=ft_params.get("ft_act"),name="shared_fc_"+str(i),kernel_regularizer=reg)(prev)
        prev=Dropout(ft_params.get("reg").get("dropout"))(prev)
    
    m=Model(fe_ins,prev)
    return(m)
    
def task_specific_logisReg(tf_out,archi_specific,nt,tnames,embed_size,conc=True,plasticity=False):
    reg=get_regularizer(archi_specific.get("reg").get("regtype"),archi_specific.get("reg").get("regparam"))
    l_out_spec=[]
    for i in range(nt):
        prevspec=tf_out.output
        for j in range(archi_specific.get("nl")):
            prevspec=Dense(archi_specific.get("nn")[j],activation=archi_specific.get("h_activation"),name="specific_fc_"+str(j)+"_for_"+str(tnames[i]),kernel_regularizer=reg)(prevspec)
            prevspec=Dropout(archi_specific.get("reg").get("dropout"))(prevspec)
        
        outspec=Dense(1,activation=archi_specific.get("o_activation"),name=str(tnames[i]))(prevspec)
        l_out_spec.append(outspec)
    
    if plasticity:
        ### We can select only some environmental variables here as well because those are sepaate models that share just the FE components ###
        beta=Dense(embed_size,activation=archi_specific.get("h_activation"),name="env_assoc_weights",kernel_regularizer=reg,use_bias=archi_specific.get('use_bias'))(prevspec)
        betassoc=Model(tf_out.input,beta)
    else:
        print("Associations are constant in environmental space.")
        betassoc=None
        
    if (conc): 
        output=Concatenate()(l_out_spec)
    else:
        output=l_out_spec
        
    hsm=Model(tf_out.input,output)
    
    return hsm,betassoc
    
#### Unit test ####  
#def ut_full_model():
#   spat_params={
#   "nbchannels":9,
#   "imsize":21,
#   "input_dropout":0.8,
#   "nbalt":2,
#   "conv":{
#       "nbfilt":[10,5],
#       "fsize":[3,3],
#       "cs":[1,1],
#       "cp":['valid','same']
#       },
#   "pool":{
#       "psize":[5,5],
#       "ps":[2,2]
#           },
#   "activ":[None,None],  ##Non-linear activation after pooling (less computation)
#   "fc":{
#       "nbfc":1,
#       "nnfc":[16],
#       "actfc":"relu"
#        },
#    "reg":{
#       "regtype":["l2","l2"],
#       "regparam":[[0.01],[0.01]],
#       "dropout":[0.8,1,0.7]
#       }}
#   spm=spatial_fe_comp(spat_params,"test")
#        
#   cat_param={"nbc":3,
#               "nmod":[10,5,26],  ### number of modalities of each variable => consider reducing PARMASE and PARMADO
#               "embed_size":[3,1,7],
#               "reg":{"regtype":"l2",
#                      "regparam":[0.01]
#                      }
#               }
#               
#   embm=cat_emb_comp(cat_param,["C1","C2","C3"])
#   num_param={ "nbnum":10,
#                "nl":1,  ##includes the hidden layers + output
#                "nn":[5],
#                "num_act":"relu",
#                "reg":{
#                   "regtype":["l2","l2"],
#                   "regparam":[[0.01],[0.01]],
#                   "dropout":0.7
#                   }}
#   numm=num_fc_comp(num_param,"num")    
#   
#   ###Here we select the feature components to use
#   fe_comps=[spm,embm,numm]
#
#   ft_params={"nl":2,  ##includes the hidden layers + output
#                   "nn":[64,32],
#                   "ft_act":"relu",
#                    "reg":{
#                       "regtype":["l2","l2"],
#                       "regparam":[[0.01],[0.01]],
#                       "dropout":0.7
#                       }}
#   ft_out=feat_transf_comp(ft_params,fe_comps)  
#    
#   archi_specific={
#            "nl":0, ##only the hidden layer, output is obviously logictic regression
#            "nn":[],
#            "h_activation":"relu",
#            "o_activation":"sigmoid",
#            "reg":{
#            "regtype":"l2",
#            "regparam":[0.01],
#            "dropout":0.7}
#           } 
#   out=task_specific_logisReg(ft_out,archi_specific,3,["A","B","C"])
#   plot_model(out)
#
#ut_full_model()
