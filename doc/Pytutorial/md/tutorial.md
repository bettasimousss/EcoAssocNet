# Tutorial on using EcoAssocNet 

This tutorials shows how to use the package on an example dataset to learn associations.

The following packages are required:
    - Data manipulation: Pandas, numpy
    - Plotting: matplotlib, seaborn
    - Machine learning: scikit-learn, tensorflow 1.5, keras 

## Part one: preparing the data

We offer a helper class DataPrep to automate data preprocessing, particularly that of environmental features.


```python
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
```

### Loading dataset 

The data used here is provided as part of the examples folder.
The data was obtained from ade4 (R package), it was produced as part of a paper from Choler et al 2005, provided within the examples/doc folder.


```python
folder_data="../examples/Aravo/data/"
file_env=folder_data+"env.csv"
file_count=folder_data+"occur.csv"
```


```python
env=pd.read_csv(file_env,sep=";",decimal=".")
env.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Form</th>
      <th>PhysD</th>
      <th>ZoogD</th>
      <th>Snow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>50</td>
      <td>no</td>
      <td>140</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>3</td>
      <td>40</td>
      <td>no</td>
      <td>140</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>20</td>
      <td>no</td>
      <td>140</td>
    </tr>
    <tr>
      <td>3</td>
      <td>9</td>
      <td>30</td>
      <td>3</td>
      <td>80</td>
      <td>no</td>
      <td>140</td>
    </tr>
    <tr>
      <td>4</td>
      <td>9</td>
      <td>5</td>
      <td>1</td>
      <td>80</td>
      <td>no</td>
      <td>140</td>
    </tr>
  </tbody>
</table>
</div>




```python
counts=pd.read_csv(file_count,sep=";",decimal=".")
names=counts.columns.tolist()
occur=(counts>0).astype(int)
counts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Agro.rupe</th>
      <th>Alop.alpi</th>
      <th>Anth.nipp</th>
      <th>Heli.sede</th>
      <th>Aven.vers</th>
      <th>Care.rosa</th>
      <th>Care.foet</th>
      <th>Care.parv</th>
      <th>Care.rupe</th>
      <th>Care.semp</th>
      <th>Fest.laev</th>
      <th>Fest.quad</th>
      <th>Fest.viol</th>
      <th>Kobr.myos</th>
      <th>Luzu.lute</th>
      <th>Poa.alpi</th>
      <th>Poa.supi</th>
      <th>Sesl.caer</th>
      <th>Alch.pent</th>
      <th>Alch.glau</th>
      <th>Alch.vulg</th>
      <th>Andr.brig</th>
      <th>Ante.carp</th>
      <th>Ante.dioi</th>
      <th>Arni.mont</th>
      <th>Aste.alpi</th>
      <th>Bart.alpi</th>
      <th>Camp.sche</th>
      <th>Card.alpi</th>
      <th>Cera.stri</th>
      <th>Cera.cera</th>
      <th>Leuc.alpi</th>
      <th>Cirs.acau</th>
      <th>Drab.aizo</th>
      <th>Drya.octo</th>
      <th>Erig.unif</th>
      <th>Gent.camp</th>
      <th>Gent.acau</th>
      <th>Gent.vern</th>
      <th>Geum.mont</th>
      <th>...</th>
      <th>Hier.pili</th>
      <th>Homo.alpi</th>
      <th>Leon.pyre</th>
      <th>Ligu.muto</th>
      <th>Lloy.sero</th>
      <th>Minu.sedo</th>
      <th>Minu.vern</th>
      <th>Phyt.orbi</th>
      <th>Plan.alpi</th>
      <th>Poly.vivi</th>
      <th>Pote.aure</th>
      <th>Pote.cran</th>
      <th>Pote.gran</th>
      <th>Puls.vern</th>
      <th>Ranu.kuep</th>
      <th>Sagi.glab</th>
      <th>Sali.herb</th>
      <th>Sali.reti</th>
      <th>Sali.retu</th>
      <th>Sali.serp</th>
      <th>Saxi.pani</th>
      <th>Sedu.alpe</th>
      <th>Semp.mont</th>
      <th>Sene.inca</th>
      <th>Sibb.proc</th>
      <th>Sile.acau</th>
      <th>Thym.poly</th>
      <th>Vero.alpi</th>
      <th>Vero.alli</th>
      <th>Vero.bell</th>
      <th>Myos.alpe</th>
      <th>Tara.alpi</th>
      <th>Scab.luci</th>
      <th>Anth.alpe</th>
      <th>Oxyt.camp</th>
      <th>Oxyt.lapp</th>
      <th>Lotu.alpi</th>
      <th>Trif.alpi</th>
      <th>Trif.badi</th>
      <th>Trif.thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>



### Preprocessing environmental data


```python
import sys
sys.path.append('../../')
```


```python
from ecoassocnet.Util.DataPrep import DataPrep
```

    Using TensorFlow backend.
    


```python
num_vars=['Slope','PhysD','Snow']
cat_vars=['Aspect','Form','ZoogD']
prep=DataPrep(num_std=["minmax"]*len(num_vars),cat_trt="onehot")
```


```python
prep.load_dataset(feat=env,occur=occur,num=num_vars,cat=cat_vars)
```


```python
prep.preprocess_numeric()
prep.process_categoric()
prep.combine_covariates()
```


```python
prep.covariates.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Slope</th>
      <th>PhysD</th>
      <th>Snow</th>
      <th>Aspect_0</th>
      <th>Aspect_1</th>
      <th>Aspect_2</th>
      <th>Aspect_3</th>
      <th>Aspect_4</th>
      <th>Aspect_5</th>
      <th>Aspect_6</th>
      <th>Aspect_7</th>
      <th>Form_0</th>
      <th>Form_1</th>
      <th>Form_2</th>
      <th>Form_3</th>
      <th>Form_4</th>
      <th>ZoogD_0</th>
      <th>ZoogD_1</th>
      <th>ZoogD_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.057143</td>
      <td>0.625</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.000000</td>
      <td>0.500</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.000000</td>
      <td>0.250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.857143</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.142857</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Part Two: Training the model

### Habitat Suitability Model pretraining (Optional)


```python
perfs,params=prep.pretrain_glms()
```


```python
fig, ax=plt.subplots(1,1,figsize=(10,20))
perfs[0].plot.barh(x='species',y='score',ax=ax,title='Area Under the Curve scores')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1bba4de59c8>




![png](output_22_1.png)



```python
biases=np.array(params[0]['b'])
weights=np.concatenate([biases,params[0]['w']],axis=1)
weights_df=pd.DataFrame(data=weights,columns=['bias']+prep.covariates.columns.tolist())
```


```python
for c in prep.groups.keys():
    if len(prep.groups[c])>1:
        weights_df[c]=weights_df[prep.groups[c]].sum(axis=1)
```


```python
fig, ax=plt.subplots(1,1,figsize=(10,10))
weights_df[num_vars+cat_vars].plot.box(ax=ax,title='Variable importance estimated by regression coefficients')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1bba66b5c08>




![png](output_25_1.png)


### Training, validation data


```python
prep.train_test_split(meth="stratified",prob=0.8)
X_train=prep.covariates.iloc[prep.idx_train,:].values
X_test=prep.covariates.iloc[prep.idx_test,:].values

Y_train=counts.iloc[prep.idx_train,:].values
Y_test=counts.iloc[prep.idx_test,:].values
```

### Setting up  configuration files


```python
from ecoassocnet.EcoAssoc import EcoAssoc, load_default_config
```


```python
from ecoassocnet.Util.Util import avgnz
```

Computing the offset to be used


```python
offsets=avgnz(counts)
```
Use the helper script 'prepare_mhsm_params.py' or modify the given sample_hsm_archi.json

```python
training_config_file='../examples/Aravo/config/association_learning.ini'
```

To understand the use of each of the following parameters, refer to the documented default config file.


```python
conf=load_default_config(training_config_file)
for k in conf.keys():
    print("%s = %s" %(k,conf[k]))
```

    exposure = True
    use_covariates = True
    intercept = False
    fixedoccur = False
    w_sigma2 = 1.0
    archi_desc_file = examples/Aravo/config/hsm_archi.json
    archi_plot_file = examples/Aravo/archi.png
    plot = False
    bias = True
    offset = False
    dist = negbin
    assoc_plasticity = False
    k = 4
    use_reg = True
    ar_sigma2 = 1.0
    prior = lasso
    lambda_lasso = 0.1
    use_penalty = False
    emb_initializer = uniform
    fixed_rho = False
    optim = sgd
    sample_ratio = 0.2
    use_valid = True
    lr = 0.01
    use_decay = False
    lr_update_step = 10000
    lr_update_scale = 0.5
    batch_size = 1
    max_iter = 5000
    nprint = 1000
    


```python
ecoasso_model=EcoAssoc(config=training_config_file,labels=names,name_dataset="aravo",target="count")
```

Hereafter, we launch the training for a few epochs (5)


```python
logg= ecoasso_model.train_interaction_model(dataset=(X_train,Y_train),verbose=1,init_weights=weights,offset=offsets)
```

    Splitting train and validation sets
    Computation graph creation (static)
    Computation graph initialization
    Setting pretrained HSM weights
    Begin training
    iteration[ 1000 ]: average llh, obj, and valid_llh are  [-35.1035095   35.92741251 -37.04162254]
    iteration[ 2000 ]: average llh, obj, and valid_llh are  [-32.04687284  32.82447647 -35.12422829]
    iteration[ 3000 ]: average llh, obj, and valid_llh are  [-31.40496827  32.15929597 -34.58151703]
    iteration[ 4000 ]: average llh, obj, and valid_llh are  [-30.35970771  31.07175282 -33.64297371]
    iteration[ 5000 ]: average llh, obj, and valid_llh are  [-29.98409594  30.65965316 -33.92271309]
    

### Evaluation

Here, we show how to evaluate a trained model given a test set


```python
perf_hsm, perf_im=ecoasso_model.evaluate_model(testdata=(X_test,Y_test))
```

    Building graph...
    Initializing...
    Calculating llh of instances...
    


```python
print("Performance of HSM component" , perf_hsm)
```

    Performance of HSM component {'microauc': 0.903777246727267}
    


```python
print("Performance of abundance component" , perf_im['pos_deviance'])
```

    Performance of abundance component 0.42921730266340535
    

### Usage for conditional predictions of abundance


```python
t=0
C=[x for x in range(len(names)) if x!=t]
Yc=Y_test[:,C]
habsuit,avg_abund=ecoasso_model.predict(X_test,C,Yc,t)
print('Predicting abundance of %s' % names[t])
(habsuit>0.5)*avg_abund
```

    Predicting abundance of Agro.rupe
    




    array([1.03985339, 1.03923207, 1.09525126, 1.05039894, 1.06305322,
           1.05340252, 1.06702837, 1.08231762, 1.05621757, 1.07785998,
           1.08209483, 1.05161302, 1.04389093, 0.        , 0.        ,
           1.00571548, 1.01322025, 0.99395305])



### Unraveling associations 


```python
from ecoassocnet.Util.Util import plot_association, plot_assoc_clustered
```


```python
assoc_df=ecoasso_model.compute_associations(save=False,norm=True)
```

We ignore intraspecific associations estimated. 


```python
assoc_df*=(np.identity(len(names))==0)
```

### Applying biogeographic filtering


```python
from ecoassocnet.Util.Util import cooccur, response_sim, biogeo_filter, plot_dendrograms
```


```python
cooc=cooccur(occur,names)
respsim=response_sim(weights)
sel_assoc=biogeo_filter(assoc_df,cooc,respsim,thoccur=0,thass=0.5,thresp=0.5,m=len(names))
```


```python
sel_assoc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Agro.rupe</th>
      <th>Alop.alpi</th>
      <th>Anth.nipp</th>
      <th>Heli.sede</th>
      <th>Aven.vers</th>
      <th>Care.rosa</th>
      <th>Care.foet</th>
      <th>Care.parv</th>
      <th>Care.rupe</th>
      <th>Care.semp</th>
      <th>Fest.laev</th>
      <th>Fest.quad</th>
      <th>Fest.viol</th>
      <th>Kobr.myos</th>
      <th>Luzu.lute</th>
      <th>Poa.alpi</th>
      <th>Poa.supi</th>
      <th>Sesl.caer</th>
      <th>Alch.pent</th>
      <th>Alch.glau</th>
      <th>Alch.vulg</th>
      <th>Andr.brig</th>
      <th>Ante.carp</th>
      <th>Ante.dioi</th>
      <th>Arni.mont</th>
      <th>Aste.alpi</th>
      <th>Bart.alpi</th>
      <th>Camp.sche</th>
      <th>Card.alpi</th>
      <th>Cera.stri</th>
      <th>Cera.cera</th>
      <th>Leuc.alpi</th>
      <th>Cirs.acau</th>
      <th>Drab.aizo</th>
      <th>Drya.octo</th>
      <th>Erig.unif</th>
      <th>Gent.camp</th>
      <th>Gent.acau</th>
      <th>Gent.vern</th>
      <th>Geum.mont</th>
      <th>...</th>
      <th>Hier.pili</th>
      <th>Homo.alpi</th>
      <th>Leon.pyre</th>
      <th>Ligu.muto</th>
      <th>Lloy.sero</th>
      <th>Minu.sedo</th>
      <th>Minu.vern</th>
      <th>Phyt.orbi</th>
      <th>Plan.alpi</th>
      <th>Poly.vivi</th>
      <th>Pote.aure</th>
      <th>Pote.cran</th>
      <th>Pote.gran</th>
      <th>Puls.vern</th>
      <th>Ranu.kuep</th>
      <th>Sagi.glab</th>
      <th>Sali.herb</th>
      <th>Sali.reti</th>
      <th>Sali.retu</th>
      <th>Sali.serp</th>
      <th>Saxi.pani</th>
      <th>Sedu.alpe</th>
      <th>Semp.mont</th>
      <th>Sene.inca</th>
      <th>Sibb.proc</th>
      <th>Sile.acau</th>
      <th>Thym.poly</th>
      <th>Vero.alpi</th>
      <th>Vero.alli</th>
      <th>Vero.bell</th>
      <th>Myos.alpe</th>
      <th>Tara.alpi</th>
      <th>Scab.luci</th>
      <th>Anth.alpe</th>
      <th>Oxyt.camp</th>
      <th>Oxyt.lapp</th>
      <th>Lotu.alpi</th>
      <th>Trif.alpi</th>
      <th>Trif.badi</th>
      <th>Trif.thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Agro.rupe</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.566786</td>
      <td>0.627253</td>
      <td>0.000000</td>
      <td>0.590266</td>
      <td>-0.000000</td>
      <td>0.725712</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.708548</td>
      <td>0.928983</td>
      <td>0.000000</td>
      <td>0.632054</td>
      <td>0.757443</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.503309</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.833124</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.813683</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.998225</td>
      <td>-0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.643385</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.777582</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.533551</td>
      <td>-0.852919</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.588186</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.841409</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.890490</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.906849</td>
      <td>0.772819</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.749003</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.925480</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.535704</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.857228</td>
      <td>-0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <td>Alop.alpi</td>
      <td>-0.771278</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.827719</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.857204</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.714540</td>
      <td>0.000000</td>
      <td>-0.00000</td>
      <td>0.567577</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.950733</td>
      <td>...</td>
      <td>0.552333</td>
      <td>0.774719</td>
      <td>0.923201</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.859838</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.975043</td>
      <td>-0.000000</td>
      <td>0.841252</td>
      <td>-0.0</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.700445</td>
      <td>-0.000000</td>
      <td>0.612286</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.666880</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.533401</td>
      <td>0.586949</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.0</td>
      <td>-0.637069</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Anth.nipp</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.509139</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.699900</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.533521</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.518235</td>
      <td>-0.761110</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.784611</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.552197</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.00000</td>
      <td>0.921969</td>
      <td>-0.740692</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.925921</td>
      <td>0.501592</td>
      <td>0.870177</td>
      <td>0.533991</td>
      <td>...</td>
      <td>0.822330</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.507264</td>
      <td>-0.000000</td>
      <td>0.737324</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.507026</td>
      <td>-0.000000</td>
      <td>0.632388</td>
      <td>-0.0</td>
      <td>-0.517872</td>
      <td>0.000000</td>
      <td>0.703382</td>
      <td>-0.600076</td>
      <td>0.573874</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.842953</td>
      <td>-0.000000</td>
      <td>0.588246</td>
      <td>-0.000000</td>
      <td>0.520055</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.797199</td>
      <td>0.700220</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.828475</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <td>Heli.sede</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.0</td>
      <td>0.858012</td>
      <td>0.571048</td>
      <td>0.512191</td>
      <td>0.928285</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.600969</td>
      <td>-0.860342</td>
      <td>0.975966</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.738068</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.732108</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.0</td>
      <td>0.577958</td>
      <td>0.650074</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.537029</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.545036</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.839731</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.678882</td>
      <td>-0.813139</td>
      <td>-0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.888695</td>
    </tr>
    <tr>
      <td>Aven.vers</td>
      <td>0.571304</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.0</td>
      <td>0.759050</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.859494</td>
      <td>0.656164</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.646750</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.770316</td>
      <td>0.871922</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.723681</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.00000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.643190</td>
      <td>0.000000</td>
      <td>0.658056</td>
      <td>-0.000000</td>
      <td>0.630899</td>
      <td>0.586907</td>
      <td>-0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>-0.611738</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.763690</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.618698</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.804011</td>
      <td>0.761456</td>
      <td>0.000000</td>
      <td>-0.638112</td>
      <td>-0.776379</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.806101</td>
      <td>-0.000000</td>
      <td>0.599935</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>Oxyt.lapp</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.652911</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.0</td>
      <td>-0.784652</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.644353</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.646418</td>
      <td>-0.602147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.710575</td>
      <td>-0.000000</td>
      <td>...</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.695468</td>
      <td>-0.000000</td>
      <td>0.643905</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.659697</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.629563</td>
      <td>-0.000000</td>
      <td>0.650538</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.668218</td>
      <td>-0.0</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Lotu.alpi</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.969321</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.592958</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.719709</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.701418</td>
      <td>-0.000000</td>
      <td>-0.953883</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.749173</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.758431</td>
      <td>0.00000</td>
      <td>0.554645</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.669008</td>
      <td>-0.554427</td>
      <td>-0.652010</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.593864</td>
      <td>0.609374</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.610612</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.580382</td>
      <td>0.000000</td>
      <td>-0.0</td>
      <td>-0.671026</td>
      <td>-0.923940</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.755450</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.878173</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.606419</td>
      <td>0.596023</td>
    </tr>
    <tr>
      <td>Trif.alpi</td>
      <td>-0.000000</td>
      <td>0.965686</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.599072</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.801169</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.849920</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.526285</td>
      <td>-0.000000</td>
      <td>-0.753277</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.722723</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.713602</td>
      <td>-0.000000</td>
      <td>0.00000</td>
      <td>0.502284</td>
      <td>-0.641366</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.541829</td>
      <td>0.000000</td>
      <td>0.756223</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.938395</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.754048</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.859597</td>
      <td>-0.000000</td>
      <td>0.710661</td>
      <td>-0.0</td>
      <td>-0.714887</td>
      <td>-0.000000</td>
      <td>0.717694</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.733401</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.865141</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.795304</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>-0.617901</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Trif.badi</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.544881</td>
      <td>0.513749</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.583669</td>
      <td>-0.000000</td>
      <td>0.0</td>
      <td>0.886880</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.649341</td>
      <td>0.000000</td>
      <td>-0.650473</td>
      <td>-0.000000</td>
      <td>-0.723502</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.745949</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.785736</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.91892</td>
      <td>0.000000</td>
      <td>-0.652394</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.739225</td>
      <td>0.527584</td>
      <td>0.991231</td>
      <td>0.607071</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.659277</td>
      <td>-0.954571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.539620</td>
      <td>-0.0</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.844513</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.869345</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.922612</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <td>Trif.thal</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.804835</td>
      <td>0.000000</td>
      <td>-0.540514</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.656606</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.687367</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.756226</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.507557</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.00000</td>
      <td>0.886383</td>
      <td>-0.700938</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.529231</td>
      <td>0.817377</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.555708</td>
      <td>-0.000000</td>
      <td>0.673132</td>
      <td>-0.930647</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.665979</td>
      <td>0.574781</td>
      <td>-0.0</td>
      <td>-0.534463</td>
      <td>0.000000</td>
      <td>0.686720</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.773136</td>
      <td>0.776484</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.593224</td>
      <td>-0.594169</td>
      <td>0.859545</td>
      <td>-0.000000</td>
    </tr>
  </tbody>
</table>
<p>82 rows × 82 columns</p>
</div>



# Part Three: Association analysis

Hereafter, we show how to analyze the learnt association matrix. We illustrate on the final association matrix (obtained after full training). Particularly, we analyze the similarities in terms of associations by performing a (hierarchical) co-clustering of the association matrix. 


```python
assoc_df=pd.read_csv('../examples/Aravo/results/plant_associations.csv',sep=";",decimal=".",index_col=0)
```


```python
g=plot_assoc_clustered(assoc_df,file=None)
```


![png](output_59_0.png)


To retrieve the species clusters shown in the figure, use the object returned by the previous function and pass it to read_dendrogram as follow:


```python
labcol=[names[x] for x in g.dendrogram_col.reordered_ind]
labrow=[names[x] for x in g.dendrogram_row.reordered_ind]
```


```python
_, _, fig=plot_dendrograms(g=g,labx='Response groups',laby='Effect groups',names=names)
fig
```




![png](output_62_0.png)




![png](output_62_1.png)



```python
fig.savefig('../examples/Aravo/results/hierarchies.pdf',bbox_inches='tight')
```
