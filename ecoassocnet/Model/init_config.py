"""
Created on Mon Apr 15 17:33:30 2019

@author: Sara Si-moussi - sara.simoussi@gmail.com
"""

import json
from .mhsm import spatial_fe_comp, cat_emb_comp, num_fc_comp, feat_transf_comp, task_specific_logisReg
from ..Util.Util import plot_archi_hsm

def init_model_from_paramfile(param_file, model_plot_file,embed_size,plot=False,plasticity=False):
    with open(param_file, 'r') as f:
        params = json.load(f)
        
    fe_comp=params["FE_comp"]
    ft_comp=params["FT_comp"]
    reg_comp=params["PRED_comp"]
    
    ##Number of feature extraction components
    nb_fe_c=len(fe_comp)
    fe_models=[]
    input_types=[]
    for fec in range(nb_fe_c):
        fe_type=fe_comp[fec][1]
        if(fe_type=="S"):
           fe_model=spatial_fe_comp(fe_comp[fec][0],fe_comp[fec][2]) 
           fe_models.append(fe_model)
           input_types.append("float32")
        elif(fe_type=="C"):
           fe_model=cat_emb_comp(fe_comp[fec][0],fe_comp[fec][2]) 
           fe_models.append(fe_model)
           input_types+=["int32"]*len(fe_comp[fec][2])
        elif(fe_type=="N"):
           fe_model=num_fc_comp(fe_comp[fec][0],fe_comp[fec][2])
           fe_models.append(fe_model)
           input_types.append("float32")
        else:
            print("Unsupported feature type ! Ignored feature.")
    
    ft_out=feat_transf_comp(ft_comp,fe_models)  
    hsm,betassoc=task_specific_logisReg(tf_out=ft_out,archi_specific=reg_comp,nt=params["NB_TASKS"],tnames=params["TASK_NAMES"],embed_size=embed_size,conc=params["CONCAT"],plasticity=plasticity) 
    if(plot):
        plot_archi_hsm(hsm,file=model_plot_file)
#        try:
#            plot_model(betassoc,to_file=model_plot_file,show_shapes=True)
#        except AttributeError:
#            pass

    return input_types, hsm, betassoc

'''
Usage

it, model, betassoc = init_model_from_paramfile(param_file,model_output_file)
'''           
