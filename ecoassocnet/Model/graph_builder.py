"""
Created on Mon Apr 15 17:33:30 2019

@author: Sara Si-moussi - sara.simoussi@gmail.com
Modified from Liping Liu repository: https://github.com/blei-lab/zero-inflated-embedding
With permission
"""

import tensorflow as tf
import numpy as np
from .init_config import init_model_from_paramfile
from keras.models import Model

class GraphBuilder:
    def __init__(self): 
        
        '''
        The computation graph contains the following variables (tensors)
        Related to the association model
        * alpha: effect embedding
        * rho: response embedding
        * invmu:  bias (or offset if fixed) of abundance in the association model
        * std: standard deviation of normal distribution 
        * intercept: log-odds of habitat suitability (for homogeneous environments, i-e: in the absence of environmental attributes.) 
        * nbr: number of trials of negative binomial (not used if other distributions are used. 
        This could be used either as a hyperparameter fixed before training or as a parameter trained along with other parameters.
    
        Related to the habitat suitability model
        * hsm: weight matrix of the habitat suitability model if used
        
        Related to associations dependence on environment
        * betassoc: contains the loadings of context effects with respect to the environmental attributes.
        They specifically contain the parameters of a regression model (Keras based) that maps for each species environmental attributes Site x Att 
        to the context representation of dimension Site x Embedding_Dimension 
        (See Supplementary Materials for details on model formulation)
        
        The HSM and BETASSOC models are populated from dedicated config files (use prepare_mhsm_params.py for assistance).
        
        As of version 0.0.1 
        - The same attributes of HSM are used in the betassoc component. 
        
        Input placeholders
        * input_att: attributes used for habitat suitability
        * input_ind: indices of positive examples (present taxa)
        * input_label: abundance (or 1) corresponding to previous indices
        '''
        
        self.alpha = None
        self.rho = None

        self.invmu = None
        self.std = None
        self.intercept= None
        #self.weight = None   ### Replaced with HSM parameters for more generality

        self.nbr = None

        self.input_att = None
        self.input_ind = None
        self.input_label = None
        
        self.hsm = None
        self.betassoc = None   

        self.debug = []  ### used for debugging 
        

    def logprob_nonz(self, alpha_emb, config, occur, training=True):
        
        idx_nonzeros_inp=tf.cast(tf.where(tf.not_equal(self.input_ind,-1)),tf.int32)
        idx_row=idx_nonzeros_inp[:,0]
        idx_col=tf.gather_nd(self.input_ind,idx_nonzeros_inp)
        
        rate = tf.cast(self.input_label, tf.float32)
        rho_select = tf.gather(self.rho, self.input_ind)
        invmu_select=tf.gather(self.invmu, self.input_ind, axis=0)
        
        # binomial distribution
        emb = tf.reduce_sum(rho_select * alpha_emb, reduction_indices=2)
        if(config['bias']==True): 
            ##Add a bias term to the natural parameter => indicates demographic traits that control abundance of the species besides ecological interactions
            emb=tf.add(invmu_select,emb)
        if config['dist'] == 'binomial': ##p=sigmoid(emb)
            logminusprob = - tf.nn.softplus(emb) ###log(1-p)
            logplusprob = - tf.nn.softplus(- emb) ###log(p)
            logprob_nz = np.log(np.math.factorial(config['bin_n'])) - self.gammaln(rate + 1.0) - self.gammaln(config['bin_n']+1 - rate) + rate * logplusprob + (config['bin_n'] - rate) * logminusprob
            logprob_z  = config['bin_n'] * logminusprob 
            log_mean = logplusprob 

        elif config['dist'] == 'poisson':
            lamb = tf.nn.softplus(emb) + 1e-6   ##Approximates the ReLu (max(0,x))
            logprob_nz = - self.gammaln(rate + 1.0) + rate * tf.log(lamb) - lamb   #### -log(y!) + y*log(lamb) - lamb / y!=Gamma(y+1) for y in N 
            logprob_z = - lamb 
            log_mean = tf.log(lamb)

        elif config['dist'] == 'negbin':
            nbr_select = tf.gather(self.nbr, self.input_ind,axis=0)
            mu = tf.nn.softplus(emb) + 1e-6
            logprob_nz = self.gammaln(rate + nbr_select) - self.gammaln(rate + 1.0) -  self.gammaln(nbr_select) + \
                         nbr_select * tf.log(nbr_select) + rate * tf.log(mu) - (nbr_select + rate) * tf.log(nbr_select + mu)

            logprob_z = nbr_select * tf.log(nbr_select) - nbr_select * tf.log(nbr_select + mu)
            log_mean = tf.log(mu)

        else:
            raise Exception('The distribution "' + config['dist'] + '" is not defined in the model')
        
        if config['exposure']:
            occur_select=tf.gather_nd(occur, tf.stack([idx_row,idx_col],axis=1))
            logits=tf.reshape(occur_select,tf.shape(emb))
            log_obs_prob = - tf.nn.softplus(- logits)   #### Log of probability of presence
            logprob = log_obs_prob + logprob_nz
        else:
            logprob = tf.expand_dims(logprob_nz,axis=0)
        
        ###Correct for padding when biotic contexts are of different sizes ###
        llh_correction=tf.cast(tf.not_equal(self.input_label,0),dtype=tf.float32) 
        logprob=logprob*llh_correction
        logprob_nz=logprob_nz*llh_correction
        logprob_z=logprob_z*llh_correction
        log_mean=log_mean*llh_correction
        
        return logprob, logprob_nz, logprob_z, log_mean


    def logprob_zero(self, context_emb, config, occur, training):
        
        movie_size = int(self.rho.get_shape()[0])
        batch_size= config['batch_size'] 
        
        # get all indices of true nonzeros
        idx_nonzeros_inp=tf.cast(tf.where(tf.not_equal(self.input_label,0)),tf.int32)
        idx_row=idx_nonzeros_inp[:,0]
        idx_col=tf.gather_nd(self.input_ind,idx_nonzeros_inp)
        
        idx_nz= idx_row*movie_size + idx_col
      
        flag = tf.scatter_nd(indices=tf.expand_dims(idx_nz,axis=1), updates=tf.tile([True],tf.shape(idx_nz)),shape=(batch_size*movie_size,))
        mask = tf.reshape(flag,[batch_size,movie_size])
        sind = tf.where(tf.equal(mask,False))
        
        if training: # if training, then subsample sind
            nsample = tf.cast(config['sample_ratio'] * tf.cast(tf.shape(sind)[0], dtype=tf.float32), tf.int32)
            sind = tf.gather(tf.random_shuffle(sind), tf.range(nsample))
        
        temp=tf.gather(context_emb,sind[:,0])
        ctx_zeros=tf.squeeze(temp,axis=1) ##axis=1 because embedding is sized 1xK
        rho_z = tf.gather(self.rho, sind[:,1]) 
        
        invmu_select=tf.gather(self.invmu, sind[:,1],axis=0)
        emb = tf.reduce_sum(rho_z * ctx_zeros, reduction_indices=1) ##sum over embedding dimensions
        if(config['bias']==True): 
            ##Add a bias term to the natural parameter => indicates demographic traits that control abundance of the species besides ecological interactions
            emb=tf.add(invmu_select,emb)        
        if config['dist'] == 'binomial':
            # binomial distribution
            # p := tf.sigmoid(emb) 
            # log(1 - p)  := - tf.nn.softplus(emb)
            logprob_z  = - config['bin_n'] * tf.nn.softplus(emb)

        elif config['dist'] == 'poisson':
            # poisson distribution
            lamb_z = tf.nn.softplus(emb) + 1e-6
            logprob_z = - lamb_z 

        elif config['dist'] == 'negbin':
            nbr_z = tf.gather(self.nbr, sind[:,1])
            mu = tf.nn.softplus(emb) + 1e-6
            logprob_z = nbr_z * tf.log(nbr_z) - nbr_z * tf.log(nbr_z + mu)

        else:
            raise Exception('The distribution "' + config['dist'] + '" is not defined in the model')
    
        if config['exposure']:
            occur_z = tf.gather_nd(occur, sind)
            logits=occur_z
            log_nobs_prob = - tf.nn.softplus(logits) 
            log_obs_prob = - tf.nn.softplus(-logits) 
            logprob = self.logsumexp(log_obs_prob + logprob_z, log_nobs_prob)
    
        else:
            logprob = tf.expand_dims(logprob_z,axis=0)
    
        return logprob, sind, emb, flag, [tf.reduce_mean(logprob_z)]
    
    
    def construct_model_graph(self, reviews, config, init_model=None, training=True, offset=None, poccur=None):
        
        review_size, movie_size, dim_atts = self.get_problem_sizes(reviews, config)
        self.initialize_model(review_size, movie_size, dim_atts, config, init_model, training, offset, poccur)
        
        if(config["exposure"]):
            if(config['use_covariates']): 
                occur_probs=self.hsm([self.input_att["feat_"+str(i)] for i in range(len(dim_atts))])
                
                if(config['assoc_plasticity']):
                    beta=self.betassoc([self.input_att["feat_"+str(i)] for i in range(len(dim_atts))])
            else: 
                ### If there is no exposure component and no covariates for SDM then there is no association plasticity
                occur_probs=self.intercept
        else:
            occur_probs=tf.zeros((tf.shape(self.input_ind)[0],movie_size),tf.float32) 
        
        # number of non-zeros
        nnz=tf.reduce_sum(tf.cast(tf.not_equal(self.input_label,0),tf.float32),1,keepdims=True)
        
        #prepare embedding of context 
        rate = tf.cast(self.input_label, tf.float32)
        alpha_select = tf.gather(self.alpha, self.input_ind, name='context_alpha')
        alpha_weighted = alpha_select * tf.expand_dims(rate, 2)
        alpha_sum = tf.reduce_sum(alpha_weighted, keepdims=True, reduction_indices=1)
        asum_zero = alpha_sum / tf.maximum(tf.expand_dims(nnz,2),1) ##divide by the true number of non zeros for each sample
        asum_nonz = (alpha_sum - alpha_weighted) / tf.expand_dims(tf.maximum(nnz-1,1),1) ###When only one non-zero entry this returns 0
        
        '''
        Config['intercept'] is deprecated 
        '''
        if(config['intercept']==True): ### Abundance is also controlled by intrinsic characteristics of the species such as its reproduction rate for instance
            ### In this case we add a bias term to the context that will be multiplied by its rho vector
            asum_zero = tf.add(asum_zero,tf.ones(shape=tf.shape(asum_zero),dtype=tf.float32,name="bias"))
            asum_nonz = tf.add(asum_nonz,tf.ones(shape=tf.shape(asum_nonz),dtype=tf.float32,name="bias"))
        
        ### Here before feeding the contexts to the generator, add the weights by betassoc ###
        if config['assoc_plasticity']:
            asum_zero=asum_zero * tf.expand_dims(beta,1)
            asum_nonz=asum_nonz * tf.expand_dims(beta,1)
        
        llh_zero, sind, emb, flag, _ = self.logprob_zero(asum_zero, config, occur_probs, training)
        llh_nonz, emb_logp_nz, emb_logp_z, log_mean = self.logprob_nonz(alpha_emb=asum_nonz, occur=occur_probs, config=config, training=training)
       
        # combine logprob of single instances
        if training:
            sum_llh_zero=tf.cond(tf.equal(tf.shape(sind)[0],0),
                    true_fn=lambda: tf.cast(0,tf.float32),
                    false_fn=lambda:tf.reduce_mean(llh_zero) * tf.cast(tf.shape(sind)[0],tf.float32))
            
            sum_llh=tf.reduce_sum(llh_nonz) + sum_llh_zero

            # training does not keep llh for each entry
            ins_llh = None 
            pos_llh = None
        else:
            canevas = tf.Variable(tf.zeros([config['batch_size']*movie_size], dtype=tf.float32),validate_shape=False)
            sind_ext=sind[:,0]*movie_size+sind[:,1]
            ins_llh = tf.scatter_update(canevas, sind_ext, llh_zero)
            pos_llh = emb_logp_nz - tf.log(1 - tf.exp(emb_logp_z))
            sum_llh = tf.reduce_sum(llh_nonz) + tf.reduce_sum(llh_zero) 

        # random choose weight vectors to get a noisy estimation of the regularization term
        rsize = max(1,int(movie_size * config['sample_ratio']))
        rind = tf.random_shuffle(tf.range(movie_size))[0 : rsize]

    
        if(config['use_reg']):
            if(config['prior']=="gaussian"):  ##L2 regularization
                if(config['bias']==True):
                    reg_bias=tf.reduce_sum(tf.square(tf.gather(self.invmu,rind)))
                else:
                    reg_bias=0
                    
                if((config['exposure']==True) & (config['use_covariates']==False)&(config['fixedoccur']==False)):
                    reg_intercept=tf.reduce_sum(tf.square(tf.gather(self.intercept,rind)))
                else:
                    reg_intercept=0
                    
                regularizer = (tf.reduce_sum(tf.square(tf.gather(self.rho,   rind)))  \
                                     + tf.reduce_sum(tf.square(tf.gather(self.alpha, rind)))
                                     + reg_bias
                                     + reg_intercept) \
                                      * (0.5 * movie_size / (config['ar_sigma2'] * rsize * review_size))
                                      
            elif(config['prior']=="lasso"):  ##L1 regularization
                if(config['bias']==True):
                    reg_bias=tf.reduce_sum(tf.abs(tf.gather(self.invmu,rind)))
                else:
                    reg_bias=0
                if((config['exposure']==True) & (config['use_covariates']==False)&(config['fixedoccur']==False)):
                    reg_intercept=tf.reduce_sum(tf.abs(tf.gather(self.intercept,rind)))
                else:
                    reg_intercept=0
                
                reg_rho = tf.reduce_sum(tf.abs(tf.gather(self.rho, rind))) \
                      * (movie_size / (config['ar_sigma2'] * rsize * review_size))
                      
                reg_alpha = tf.reduce_sum(tf.abs(tf.gather(self.alpha, rind))) \
                      * (movie_size / (config['ar_sigma2'] * rsize * review_size))
                
                regularizer=config['lambda_lasso'] * (reg_rho + reg_alpha + reg_intercept + reg_bias)            
            else:
                regularizer = 0
        
        if(config['use_penalty']):
            pen=0 ##TODO: add other penalties here
        
        else:
            pen=0
            
        objective = regularizer + pen  - sum_llh   
            
        inputs = {'input_ind': self.input_ind, 'input_label': self.input_label} 
        inputs.update(self.input_att)
        outputs = {'objective': objective, 'llh': sum_llh, 'ins_llh': ins_llh, 'pos_llh': pos_llh, 'debugv': [],#[train_writer,valid_writer], 
                   'exposure':occur_probs, 'means':log_mean, 'meansneg':emb, 'sindzero':sind, 'flag':flag}
        model_param = {'alpha': self.alpha, 'rho': self.rho,  
                       'invmu': self.invmu, 'std':self.std, 'intercept':self.intercept,
                       'nbr': self.nbr, 'hsm_params': self.hsm}
        
        if config['assoc_plasticity']:
            model_param.update(dict(betassoc_params=self.betassoc))
    
        return inputs, outputs, model_param 
    

    def initialize_model(self, review_size, movie_size, dim_atts, config, init_model=None, training=True, offset=None, poccur=None):
        ###Setting up covariates and related parameters###
        if(config['exposure'] & config['use_covariates']):
            archihsm_params=config["archi_desc_file"]
            archihsm_plot=config["archi_plot_file"]
            in_types,multihsm,betassoc=init_model_from_paramfile(param_file=archihsm_params,model_plot_file=archihsm_plot,embed_size=config['k'],plot=config['plot'],plasticity=config["exposure"] and config["use_covariates"] and config["assoc_plasticity"])
            
            self.input_att = {}
            for i in range(len(dim_atts)):
                dims=dim_atts[i] 
                if(len(dims)==1):
                    dims=[None,dims[0]]
                ph=tf.placeholder(in_types[i], shape=dims)
                self.input_att["feat_"+str(i)]=ph
            
            self.hsm = multihsm
            self.betassoc=betassoc
        else:
            self.input_att={}
            self.hsm= Model()
        
        embedding_size = config['k']
        self.input_label = tf.placeholder(tf.int32, shape=[None,None],name='input_label')
        self.input_ind = tf.placeholder(tf.int32, shape=[None,None],name='input_ind')
        
        if training: 
            if init_model == None:  ###Uninformative priors
                
                if(config['emb_initializer']=="uniformsmall"):
                    self.alpha  = tf.Variable(tf.random_uniform([movie_size, embedding_size], -0.01, 0.01))
                    self.rho    = tf.Variable(tf.random_uniform([movie_size, embedding_size], -0.01, 0.01))
                elif(config['emb_initializer']=="uniform"):
                    self.alpha  = tf.Variable(tf.random_uniform([movie_size, embedding_size], -0.5, 0.5))
                    self.rho    = tf.Variable(tf.random_uniform([movie_size, embedding_size], -0.5, 0.5))
                elif(config['emb_initializer']=="glorot"):
                    initializer = tf.contrib.layers.xavier_initializer()
                    self.alpha  = tf.Variable(initializer([movie_size, embedding_size]))
                    self.rho    = tf.Variable(initializer([movie_size, embedding_size])) 
                elif(config['emb_initializer']=="truncated_normal"):
                    self.alpha  = tf.Variable(tf.truncated_normal([movie_size, embedding_size], 0, 0.5))
                    self.rho    = tf.Variable(tf.truncated_normal([movie_size, embedding_size], 0, 0.5)) 
                
                if(config['fixed_rho']):
                    self.rho = tf.Variable(tf.ones((movie_size,embedding_size)),trainable=False)
                
                if(config['offset']):
                    self.invmu  = tf.constant(offset)
                else:
                    self.invmu  = tf.Variable(tf.random_uniform([movie_size], -1, 0.5)) ##For prevalences between 0 and 0.5
                
                if(config['fixedoccur']):
                    self.intercept  = tf.constant(poccur)
                else:
                    self.intercept  = tf.Variable(tf.random_uniform([1,movie_size], -1, 0.5))
                
                self.nbr  = tf.nn.softplus(tf.Variable(tf.random_uniform([movie_size], -1, 1))) ##Softplus ensures positivity
                self.std = tf.Variable(tf.random_uniform([movie_size], 0, 10))
            else:
                self.alpha  = tf.Variable(init_model['alpha'])
                self.invmu  = tf.Variable(init_model['invmu'])
                self.intercept  = tf.Variable(init_model['intercept'])
                self.rho    = tf.Variable(init_model['rho'])
                self.std   = tf.Variable(init_model['std'])

                free_nbr = self.inv_softplus_np(init_model['nbr'])
                self.nbr  = tf.nn.softplus(tf.Variable(free_nbr))
                print('use parameters of the initial model')
        else: 
            self.alpha  = tf.constant(init_model['alpha'])
            self.invmu  = tf.constant(init_model['invmu'])
            self.intercept  = tf.constant(init_model['intercept'])
            self.rho    = tf.constant(init_model['rho'])
            self.nbr = tf.constant(init_model['nbr'])
            self.std = tf.constant(init_model['std'])
            
    
    '''
                    Supporting functions
    '''
    def get_problem_sizes(self, reviews, config):
        review_size = reviews['scores'].shape[0]
        movie_size = reviews['scores'].shape[1]
        nbf=len(reviews['atts'])
        dim_atts = [reviews['atts'][i].shape[1:] for i in range(nbf)]
        
        return review_size, movie_size, dim_atts

    def logsumexp(self, vec1, vec2):
        flag = tf.greater(vec1, vec2)
        maxv = tf.where(flag, vec1, vec2)
        lse = tf.log(tf.exp(vec1 - maxv) + tf.exp(vec2 - maxv)) + maxv
        return lse

    def gammaln(self, x):
        # fast approximate gammaln from paul mineiro
        # http://www.machinedlearnings.com/2011/06/faster-lda.html
        logterm = tf.log (x * (1.0 + x) * (2.0 + x))
        xp3 = 3.0 + x
        return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tf.log (xp3)


    def inv_softplus_np(self, x):
        y = np.log(np.exp(x) - 1)
        return y 
   
