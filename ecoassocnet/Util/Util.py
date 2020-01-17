# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score, jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from scipy.cluster import hierarchy

try:
    from keras.utils.vis_utils import plot_model
except:
    pass


#def plot_archi_hsm(hsm):
#    hsm.summary()
def plot_archi_hsm(hsm,file):
    try:
        plot_model(hsm,to_file=file,show_shapes=True)
    except OSError:
        hsm.summary()
        
    except:
        print('Uknown error')
        pass

######## Training helper functions ########
def test_convergence(llh):
    ##Test convergence
    ##If the variations are up and down => derivative is negative than positive than negative
    deriv=np.zeros((len(llh)-1,1))
    for i in range(len(deriv)):
        deriv[i]=llh[i+1]-llh[i]
    
    nbdec=len(deriv[deriv<0])
    if(nbdec>len(deriv)/2):
        cv=False  
    else:
        cv=True
    
    return cv

######## Metrics function ########
def evaluate_pairwise_assoc(pa_b,pa_nb,pb_a,pb_na):  ###Computes the various odds ratios given conditional probabilities
    ###### Probability of A ####### 
    pna_b=1-pa_b
    pna_nb=1-pa_nb
    
    IR_A=pa_b/pa_nb
    ER_A=pna_b/pna_nb
    OA_B=pa_b/pna_b
    OA_NB=pa_nb/pna_nb
    
    OR_A=IR_A/ER_A
    
    ###### Probability of B #######     
    pnb_a=1-pb_a
    pnb_na=1-pb_na
    
    IR_B=pb_a/pb_na
    ER_B=pnb_a/pnb_na
    OB_A=pb_a/pnb_a
    OB_NA=pb_na/pnb_na
    
    OR_B=IR_B/ER_B
    
    return dict(A=dict(OAB=OA_B,OANB=OA_NB,ORA=OR_A, IR_A=IR_A, ER_A=ER_A),B=dict(OBA=OB_A,OBNA=OB_NA,ORB=OR_B,IR_B=IR_B, ER_B=ER_B))

def compute_AIC(logllh, pool_size, embed_size,p=1,intercept_fit=False):
    paramsize = (2*embed_size +int(intercept_fit)+p) * pool_size
    return 2*(paramsize - logllh)

def compute_deviance(y,u,u_log=True):
    if(u_log==False):
        u=np.log(u)
    d=np.multiply(y,(np.log(y)-u)) -y + np.exp(u)
    return 2*d

def compute_AUC_from_logit(logits,counts):
    true_pres=(counts.todense()>0).astype(int)
    pred_probs=sigmoid(logits)
    
    return roc_auc_score(true_pres,pred_probs,'macro')

######## Data transformation functions ######## 
def get_scaler(num_std):
    if(num_std=="standard"):
        scaler=StandardScaler()
    elif(num_std=="minmax"):
        scaler=MinMaxScaler()
    elif(num_std=="robust"):
        scaler=RobustScaler()
    else:
        scaler=None
    
    return scaler

def sigmoid(x):
    return 1/(1+np.exp(-x))

def poly(x,p):  ###QR decomposition of the matrix of powers of x
    x=np.array(x)
    powers=np.concatenate([x**k for k in range(p+1)],axis=1)
    qr=np.linalg.qr(powers)
    return qr[0][:,1:]

def minnz(arr):
    return np.min(arr[arr>0])

def mednz(arr):
    return np.median(arr[arr>0])

def avgnz(arr):
    return np.mean(arr[arr>0])

def compute_offset(data,offset_mode):  
    if offset_mode=='minnz':
        return minnz(data)
    
    if offset_mode=='mednz':
        return None
    
    if offset_mode=='avgnz':
        return None
    
    return None

def minmx(X,maxv=1,minv=-1):
    X_std=(X - X.min().min()) / (X.max().max() - X.min().min())
    X_scaled=X_std * (maxv - minv) + minv
    return X_scaled
    
######## Post-processing associations #####################
def postprocess(fit_assoc,trt="raw",mx=1,mn=-1):
    if trt=="normresp":
        result=MinMaxScaler(feature_range=(-1,1)).fit_transform(fit_assoc.T).T
    elif trt=="normeff":
        result=MinMaxScaler(feature_range=(-1,1)).fit_transform(fit_assoc)
    else:
        result=fit_assoc
    
    return pd.DataFrame(data=result,columns=fit_assoc.columns,index=fit_assoc.index)

######## Plot functions ########
### Plot response curve, plot relative effect, plot serie of boxplots
def plot_decision(vmin,vmax,clf,title,figname):
    x1, x2 = np.mgrid[vmin:vmax:.001, vmin:vmax:.001]
    grid = np.c_[x1.ravel(), x2.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(x1.shape)
    
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(x1, x2, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    
    ax.set(aspect="equal",
           title=title,
           xlim=(-1, 1), ylim=(-1, 1),
           xlabel="$X_1$", ylabel="$X_2$")
    
    f.savefig(figname)
    plt.close()

def plot_predictions(X,y,mu,sigma,mode="sep",figout="figure.png"):
    ###Normalized probability
    y_true=stats.norm.pdf(X, mu, sigma)/(stats.norm.pdf(0)/(sigma))
    if(mode=="sep"):
        f, (ax1, ax2)=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=False)
        ax1.scatter(X,y,marker='o')
        ax2.plot(X, y_true)
        plt.savefig(figout)
    else:
        plt.scatter(X,y)
        plt.plot(X,y_true)
        plt.savefig(figout)
        plt.close()
        

def plot_association(mat,cmap='seismic_r',fileout=None,xdim=20,ydim=20,vmin=-1,vmax=1,title=""):
    ### Plots a heatmap of the association matrix
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(xdim,ydim))
    ax.title.set_text(title)
    sns.heatmap(mat,cmap=cmap,ax=ax, vmin=vmin, vmax=vmax,xticklabels=True,yticklabels=True)
    if fileout is not None:
        fig.savefig(fileout)
        plt.close()
    

def plot_confusion_matrix(cm, classes,classnames,
                       normalize=False,
                       title=None,
                       cmap=plt.cm.Blues):
 """
 This function prints and plots the confusion matrix.
 Normalization can be applied by setting `normalize=True`.
 """
 if not title:
     if normalize:
         title = 'Normalized confusion matrix'
     else:
         title = 'Confusion matrix, without normalization'
         
 if normalize:
     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
     cm[np.isnan(cm)]=0
     print("Normalized confusion matrix")
 else:
     print('Confusion matrix, without normalization')
 
 print(cm)
 
 fig, ax = plt.subplots()
 im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
 ax.figure.colorbar(im, ax=ax)
 # We want to show all ticks...
 ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classnames, yticklabels=classnames,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')
 
 # Rotate the tick labels and set their alignment.
 plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")
 
 # Loop over data dimensions and create text annotations.
 fmt = '.2f' if normalize else 'd'
 thresh = cm.max() / 2.
 for i in range(cm.shape[0]):
     for j in range(cm.shape[1]):
         ax.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
 fig.tight_layout()
 return fig
 
 
def classify(a,b,mode=0):
 if(mode==0):
     codes={0:'0',1:'+',-1:'-'}
     cl=codes[a]+'_'+codes[b]
 else:
     codes={(0,0):'0',
            (0,1):'+',(1,0):'+',(1,1):'+',
            (0,-1):'-',(-1,0):'-',(-1,-1):'-',
            (1,-1):'+-',(-1,1):'+-'}
     cl=codes[(a,b)]
 return cl

################## Diagnosis ############################
def boxplot_pred(folder_box,used_runs,names,colors,
                 df,groups,groupvar,runvar,metvar,
                 title,file_out,scale=3,bxp="all"):
    relab_traces=[]
    for gr in groups:
        data=df.query(groupvar+'==@gr & '+ runvar +' in @used_runs')[[runvar,metvar]]
        relab_traces.append(
        go.Box(
            y=data[metvar],
            name=str(gr),
            x=[names.get(rn) for rn in data[runvar].tolist()],
            marker=dict(
            color=colors.get(str(gr))
            ),
            boxpoints=bxp
        ))
        
    layout=go.Layout(
            title=go.layout.Title(
            text=title),
            boxmode='group',
            xaxis=go.layout.XAxis(
                showticklabels=True,
                showgrid=True,
                gridwidth=0.5,
                #nticks=30,
                tickangle=90
        )
    )
    
    fig=go.Figure(data=relab_traces,layout=layout)
    pio.write_image(fig,file_out,scale=scale)
    



def plot_variable_importance(vi,group_vars,names,file_out,scale=2):

    traces=[]
    x = vi.columns.tolist()
    
    for f in x:
        traces.append(
            go.Box(
                y=vi.loc[:,f],
                x=names,
                name=f,
            )
        )
            
    layout = go.Layout(
        yaxis=dict(
            title='Log odds ratio, variable importance',
            zeroline=False),
        boxmode='group'
    )
    
    fig = go.Figure(data=traces, layout=layout)
    pio.write_image(fig,file_out,scale=scale)        
    
################## Dataset analysis ############################
### Co-occurrence metrics
def cooccur(occur,species):
    m=occur.shape[1]
    jacc=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            jacc[i,j]=jaccard_score(occur.iloc[:,i],occur.iloc[:,j])
    
    return pd.DataFrame(data=jacc,columns=species,index=species)

### Response similarity ###
def response_sim(df):
    return cosine_similarity(df)


def env_overlap(df,label='species',envvar='env',occurvar='abundance'):
    spec_list=df[label].unique().tolist()
    
    over=list()
    
    for i in spec_list:
        xi=df.query(label+'==@i & '+occurvar+'>0')
        minxi=xi[envvar].min()
        maxxi=xi[envvar].max()
        for j in spec_list:
            xj=df.query(label+'==@j & '+occurvar+'>0')
            minxj=xj[envvar].min()
            maxxj=xj[envvar].max()
            
            over.append(dict(xi=i,xj=j,
                             mini=minxi,minj=minxj,
                             maxi=maxxi,maxj=maxxj,
                             overlap=(min(maxxi,maxxj)-max(minxi,minxj))/min(maxxi-minxi,maxxj-minxj)
                             )
            )
    
    overdf=pd.DataFrame.from_dict(over)  
    
    return overdf      
    

### Comparative analysis ###
def compare_pattern_fit(abund,species,list_mats,names_mats):
    #data: n*m matrix
    #index: m size list
    #list_mats: list of m*m matrices to contrast with data
    #names_mats: same size as list_mats, gives name of attributes
    m=abund.shape[1]
    relabmu=np.zeros((m,m))
    relabsd=np.zeros((m,m))
    absrelabmu=np.zeros((m,m))
    absrelabsd=np.zeros((m,m))
    
    rel_abund=[]
    for x in range(m):  ## response / target
        spx=species[x]
        target_avgcount=avgnz(abund.iloc[:,x])
        for y in range(m): ##effect / conditional
            if(y!=x):
                spy=species[y]
                xcount_y=abund[(abund[spx]>0)&(abund[spy]>0)][spx] - target_avgcount
                absxcount_y=abund[(abund[spy]>0)][spx] - target_avgcount
                
                if(len(xcount_y)==0):
                    relabmu[x,y]=0
                    relabsd[x,y]=0
                else:    
                    relabmu[x,y]=xcount_y.mean()
                    relabsd[x,y]=xcount_y.std()
                    
                if(len(xcount_y)==0):
                    absrelabmu[x,y]=0
                    absrelabsd[x,y]=0
                else:    
                    absrelabmu[x,y]=absxcount_y.mean()
                    absrelabsd[x,y]=absxcount_y.std()
                
                xyvals={'x':spx,'y':spy,#'x_y':xcount_y,
                                  'mu':relabmu[x,y],'sd':relabsd[x,y],
                                  'absmu':absrelabmu[x,y],'abssd':absrelabsd[x,y]}
                
                for k in range(len(list_mats)):
                    val=list_mats[k].loc[spx,spy]
                    key=names_mats[k]
                    xyvals.update({key:val})
                    
                rel_abund.append(xyvals)  
    
    return pd.DataFrame.from_dict(rel_abund)


def n_signif(mat,n,pos=True): ##mat should be a pandas dataframe
    fitmat_long=mat.melt(var_name="Source",value_name="Effect")
    fitmat_long['Target']=mat.columns.tolist()*mat.shape[0]   
    fitmat_long["Effect"]=fitmat_long["Effect"].astype(float)
    if(pos):
        fitmat_long=fitmat_long[fitmat_long.Effect>0]
    else:
        fitmat_long=fitmat_long[fitmat_long.Effect<0]
        
    fitmat_long.sort_values(by=["Effect"],inplace=True,ascending=not(pos))
    topn=fitmat_long.head(n)
    
    return topn


############ Bootstrap related functions ###################   
def CI(list_df,popsize):
    ###### CI96% for the mean = mean +- 1.96*SE  s.t SE=STD/sqrt(popsize), popsize=#bootstrap samples  #####
    newdf=np.stack(list_df)
    meandf=newdf.mean(axis=0)
    sedf=newdf.std(axis=0)/np.sqrt(popsize)
    ci_inf=meandf-1.96*sedf
    ci_sup=meandf+1.96*sedf
    
    return pd.DataFrame(ci_inf), pd.DataFrame(ci_sup)


def CI_product(list_a,list_b,popsize,list_ab=None):
    '''
    Using the Delta method: https://web.stanford.edu/class/cme308/OldWebsite/notes/TaylorAppDeltaMethod.pdf
    Supposes that different estimations of the embeddings can be summed (identifiable by rotation or translation)
    Do not use here
    '''
    if list_ab==None:
        list_ab=[np.dot(list_a[i],list_b[i].T) for i in range(len(list_a))]
        
    exp_ab=np.mean(np.stack(list_ab),axis=0)
    exp_a=np.mean(np.stack(list_a),axis=0)
    exp_b=np.mean(np.stack(list_b),axis=0)
    var_a=np.std(np.stack(list_a),axis=0)
    var_b=np.std(np.stack(list_b),axis=0)
    
    se_ab=np.sqrt((np.dot(exp_a*exp_a,var_b.T)+np.dot(exp_b*exp_b,var_a.T))/popsize)
    
    ci_inf=exp_ab-1.96*se_ab
    ci_sup=exp_ab+1.96*se_ab
    
    return pd.DataFrame(ci_inf), pd.DataFrame(ci_sup)
    
    
def testsignif_assoc(assoc_inf,assoc_sup):
    ### if sign(a*b)<0 => sign(a)!=sign(b) it means the confidence interval contains 0 => the corresponding cell should be set to 0
    mask=(assoc_inf*assoc_sup)<0
    
    return mask


############################### Visualization tools ########################################################
def plot_assoc_clustered(assoc,file,palette='seismic_r',row_colors=None,col_colors=None):
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "figure.titleweight" : 'bold',
        "pgf.preamble": [
             r"\usepackage[utf8x]{inputenc}",
             r"\usepackage[T1]{fontenc}",
             r"\usepackage{cmbright}",
             ]
    })
    
    g=sns.clustermap(assoc, center=0, cmap=palette,
                   row_colors=row_colors, col_colors=col_colors,
                   linewidths=.75, figsize=(30, 30))
    
    if file is not None:
        g.savefig(file,bbox_inches='tight')
        
        
    return g


def biogeo_filter(assoc_df,cooc,respsim,thoccur,thass,thresp,m):
    thass=0.5
    thresp=0
    thoccur=0

    mask_pos=(assoc_df>thass)*(cooc>thoccur)
    mask_neg=(assoc_df<-thass)*(respsim>thresp)
    mask=(mask_pos+mask_neg)
    
    return assoc_df*mask

def plot_dendrograms(g,names,labx='Response groups',laby='Effect groups'):
    labcol=[names[x] for x in g.dendrogram_col.reordered_ind]
    labrow=[names[x] for x in g.dendrogram_row.reordered_ind]
    
    fig, ax=plt.subplots(2,1,figsize=(30,20))
    dn_row = hierarchy.dendrogram(g.dendrogram_row.linkage,ax=ax[0],labels=labrow,leaf_font_size=12)
    ax[0].set_title(labx)
    dn_col = hierarchy.dendrogram(g.dendrogram_col.linkage,ax=ax[1],labels=labcol,leaf_font_size=12)
    ax[1].set_title(laby)
    
    return dn_row, dn_col, fig

#def concat_weights(l_weights):
#    return None