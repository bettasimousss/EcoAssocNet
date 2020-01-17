"""
Created on Mon Apr 15 17:33:30 2019

@author: Sara Si-moussi - sara.simoussi@gmail.com
Modified from Liping Liu repository: https://github.com/blei-lab/zero-inflated-embedding
With permission
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy import sparse
from .graph_builder import GraphBuilder

def separate_valid(reviews, frac):
    review_size = reviews['scores'].shape[0]
    vind = np.random.choice(review_size, int(frac * review_size), replace=False)
    tind = np.delete(np.arange(review_size), vind)
    
    nbf=len(reviews['atts'])
    trainset = dict(scores=reviews['scores'][tind, :], atts=[reviews['atts'][i][tind] for i in range(nbf)])
    validset = dict(scores=reviews['scores'][vind, :], atts=[reviews['atts'][i][vind] for i in range(nbf)])
    
    return trainset, validset


def validate(valid_reviews, session, inputs, outputs):
    valid_size = valid_reviews['scores'].shape[0]
    ins_llh = np.zeros(valid_size)
    for iv in range(valid_size): 
        atts, indices, labels = generate_batch(valid_reviews, [iv])

        dict_atts={}
        for i in range(len(atts)):
            dict_atts.update({inputs["feat_"+str(i)]:atts[i]})
        
        feed_dict = {inputs['input_ind']: indices.astype(np.int32), inputs['input_label']: labels.astype(np.int32)}
        feed_dict.update(dict_atts)
        
        ins_llh[iv] = session.run((outputs['llh']), feed_dict=feed_dict)
    
    mv_llh = np.mean(ins_llh)
    return mv_llh


def fit_emb(reviews, config, init_weights=None, init_emb=None, verbose=1, tb=False, offset=None, poccur=None):
    np.random.seed(27)

    # this options is only related to training speed. 
    
    if verbose:
        print("Splitting train and validation sets")

    use_valid_set = config['use_valid']
    if use_valid_set:
        reviews, valid_reviews = separate_valid(reviews, 0.1)
        
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        
        if verbose:
            print("Computation graph creation (static)")  
            
        builder = GraphBuilder()
        inputs, outputs, model_param = builder.construct_model_graph(reviews, config, init_model=init_emb, training=True, offset=offset, poccur=poccur)
        
        ###Setting up exponential decay
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = config['lr']
        
        #print("Configuration")
        if(config['use_decay']):  ###only with SGD
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   config['lr_update_step'], config['lr_update_scale'], staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(outputs['objective'],global_step=global_step)
        else:
            learning_rate=starter_learning_rate
            if(config['optim']=="adam"):
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(outputs['objective'])
            elif(config['optim']=="adadelta"):
                optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(outputs['objective'])
            elif(config['optim']=="sgd"):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(outputs['objective'])
            else:
                optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(outputs['objective'])
            
        
        if verbose: 
            print("Computation graph initialization")   
        init = tf.global_variables_initializer()
    
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        ###Reinitialize placeholders each time we recall the graph
        init.run()
        
        if(init_weights != None):
            if verbose:
                print("Setting pretrained HSM weights")
            builder.hsm.set_weights(init_weights)
                
        if(tb):  ##Tensorboard summary
            tf.summary.FileWriter("TB_LOGS",session.graph)
        
        nprint = config['nprint']
        val_accum = np.array([0.0, 0.0])
        train_logg = np.zeros([int(config['max_iter'] / nprint) + 1, 3]) 

        review_size = reviews['scores'].shape[0]
        max_llh=-1E10 ##arbitrarily big llh
        mle_model=model_param.copy() ##Initial model
        
        if verbose:
            print("Begin training")
        sinds=[]
        flags=[]
        for step in range(1, config['max_iter'] + 1):
            ####Shuffle indices
            rind = np.random.choice(review_size,config['batch_size'])
            atts, indices, labels = generate_batch(reviews, rind)
            dict_atts={}
            for i in range(len(atts)):
                dict_atts.update({inputs["feat_"+str(i)]:atts[i]})
            
            feed_dict = {inputs['input_ind']: indices.astype(np.int32), inputs['input_label']: labels.astype(np.int32)}
            feed_dict.update(dict_atts)

            _, llh_val, obj_val, debug_val, sind, flag= session.run((optimizer, outputs['llh'], outputs['objective'], outputs['debugv'], outputs['sindzero'], outputs['flag']), feed_dict=feed_dict)
            val_accum = val_accum + np.array([llh_val, obj_val])
            sinds.append(sind)
            flags.append(flag)
            
            # print loss every nprint iterations
            if step % nprint == 0 or np.isnan(llh_val) or np.isinf(llh_val):
                
                #outputs['debug'][1].add_summary(debug_val[1],step)
                
                valid_llh = 0.0
                break_flag = False
                if use_valid_set:
                    valid_llh = validate(valid_reviews, session, inputs, outputs)
                    ref_llh=valid_llh
                else:
                    ref_llh=llh_val
                
                if(ref_llh>max_llh): ##Better than the current mle
                    max_llh=ref_llh
                    mle_model = dict(alpha=model_param['alpha'].eval(), 
                           rho=model_param['rho'].eval(), 
                          invmu=model_param['invmu'].eval(), 
                          intercept=model_param['intercept'].eval(), 
                           nbr=model_param['nbr'].eval(),
                           std=model_param['std'].eval(),
                           hsm=model_param['hsm_params'].get_weights())
                    
                    if config['assoc_plasticity']:
                        mle_model.update(dict(betassoc=model_param['betassoc_params'].get_weights()))
                
                # record the three values 
                ibatch = int(step / nprint)
                train_logg[ibatch, :] = np.append(val_accum / nprint, valid_llh)
                
                val_accum[:] = 0.0 # reset the accumulater
                if(verbose==1):
                    print("iteration[", step, "]: average llh, obj, and valid_llh are ", train_logg[ibatch, :])
                
                if np.isnan(llh_val) or np.isinf(llh_val):
                    print('Loss value is ', llh_val, ', and the debug value is ', debug_val)
                    raise Exception('Bad values')
   
                if break_flag:
                    break

        # save model parameters to dict: last model and mle_model (best on validation set or best on training set so far)
        model = dict(alpha=model_param['alpha'].eval(), 
                       rho=model_param['rho'].eval(), 
                      invmu=model_param['invmu'].eval(),
                      intercept=model_param['intercept'].eval(),
                     
                       nbr=model_param['nbr'].eval(),
                       std=model_param['std'].eval(),
                       hsm=model_param['hsm_params'].get_weights())
        
        if config['assoc_plasticity']:
            model.update(dict(betassoc=model_param['betassoc_params'].get_weights()))

        return model, train_logg, mle_model, sinds, flags

def evaluate_emb(reviews, model, config):  ##in this case, model contains both HSM params and embedding params
    
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        # construct model graph
        print('Building graph...')
        builder = GraphBuilder()
        inputs, outputs, model_param = builder.construct_model_graph(reviews, config, model, training=False)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        print('Initializing...')
        init.run()
        builder.hsm.set_weights(model['hsm'])
        
        if(config['assoc_plasticity']):
            builder.betassoc.set_weights(model['betassoc'])  
            
        llh_array = [] 
        pos_llh_array = [] 
        occur_probs_array = []
        logmeans=[]
        nnz_idx=[]
        lnemb=[]
        review_size = reviews['scores'].shape[0]
        print('Calculating llh of instances...')
        for step in range(review_size):                
            att, index, label = generate_batch(reviews, [step])
            feed_dict = {inputs['input_ind']: index.astype(np.int32), inputs['input_label']: label.astype(np.int32)}
            dict_atts={}
            for i in range(len(att)):
                dict_atts.update({inputs["feat_"+str(i)]:att[i]})
                        
            feed_dict.update(dict_atts)
            
            ins_llh_val, pos_llh_val, occur_probs, log_mean, negemb, sind = session.run((outputs['ins_llh'], outputs['pos_llh'], outputs['exposure'], outputs['means'], outputs['meansneg'],outputs['sindzero']), feed_dict=feed_dict)
            
            logmeans.append(log_mean)
            lnemb.append((sind,negemb))
            nnz_idx.append(index)
            llh_array.append(ins_llh_val)
            pos_llh_array.append(pos_llh_val)
            occur_probs_array.append(occur_probs)

        llh_array = np.concatenate(llh_array, axis=0)
        probs_array=np.concatenate(occur_probs_array,axis=0)
        
        return llh_array, pos_llh_array, probs_array, logmeans, nnz_idx, builder.hsm.get_weights(), lnemb

def generate_batch(reviews, rind):
    nbf=len(reviews['atts'])
    atts = [reviews['atts'][i][rind, :] for i in range(nbf)]
    linds=[sparse.find(reviews['scores'][r, :])[1].astype(np.int32) for r in rind]
    lrates=[sparse.find(reviews['scores'][r, :])[2].astype(np.int32) for r in rind]
    lengths=[len(linds[j]) for j in range(len(rind))]
    maxl=np.max(lengths)
    minl=np.min(lengths)
    if(maxl>minl): ##Apply padding to have an homogeneous context size within batches
        to_pad=np.where(lengths<np.max(lengths))[0]
        for l in to_pad:
            pad_size=maxl-lengths[l]
            linds[l]=np.pad(linds[l],(0,pad_size),'constant', constant_values=(0))
            lrates[l]=np.pad(lrates[l],(0,pad_size),'constant', constant_values=(0))
            
    ind=np.stack(linds)
    rate=np.stack(lrates)
    
    ##Concatenate the results
    ind=np.stack(linds)
    rate=np.stack(lrates)
    return atts, ind, rate 
