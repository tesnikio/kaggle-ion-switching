import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import f1_score
import scipy
from scipy import signal

train = pd.read_csv('../input/50hz-noise-2-724-zhiraf/train_kalman_clean.csv') #version 3
test = pd.read_csv('../input/50hz-noise-2-724-zhiraf/test_kalman_clean.csv')

def gmm_feature(train, test, n_components, cov_type):
    
    means = []
    
    for i in range(n_components):
        res_mean = train[train['open_channels'] == i]['clear_signal'].mean()
        means.append(res_mean)
        
    means = np.array(means).reshape(-1,1)
    
    signal = np.hstack([train['clear_signal'], test['clear_signal']]).reshape(-1, 1)
    
    gmm = GaussianMixture(n_components = n_components, random_state = 0, covariance_type = cov_type, 
                          means_init = means, max_iter = 20000, tol = 1e-6)
    gmm.fit(signal)
    
    train_gmm = pd.DataFrame(gmm.predict_proba(train['clear_signal'].values.reshape(-1,1)))
    test_gmm = pd.DataFrame(gmm.predict_proba(test['clear_signal'].values.reshape(-1,1)))
    
    cols = []
    
    for i in range(n_components):
        res = 'gmm_' + str(i)
        cols.append(res)
    
    train_gmm.columns = cols
    test_gmm.columns = cols
    
    train = pd.concat([train,train_gmm],axis=1)
    test = pd.concat([test,test_gmm],axis=1)
    
    return train, test, cols

train, test, cols = gmm_feature(train,test, n_components = 11, cov_type = 'spherical')

train['prediction_float'] = train['gmm_0'] * 0 + train['gmm_1'] * 1 + train['gmm_2'] * 2 + train['gmm_3'] * 3 + train['gmm_4'] * 4 + train['gmm_5'] * 5 + train['gmm_6'] * 6 + train['gmm_7'] * 7 + train['gmm_8'] * 8 + train['gmm_9'] * 9 + train['gmm_10'] * 10
test['prediction_float'] = test['gmm_0'] * 0 + test['gmm_1'] * 1 + test['gmm_2'] * 2 + test['gmm_3'] * 3 + test['gmm_4'] * 4 + test['gmm_5'] * 5 + test['gmm_6'] * 6 + test['gmm_7'] * 7 + test['gmm_8'] * 8 + test['gmm_9'] * 9 + test['gmm_10'] * 10

train['prediction_round'] = np.clip(np.round(train['prediction_float']),0,10).astype(int)
test['prediction_round'] = np.clip(np.round(test['prediction_float']),0,10).astype(int)

train['argmax_pred'] = train[cols].max(axis=1)
test['argmax_pred'] = test[cols].max(axis=1)

train.loc[train['gmm_0'] == train['argmax_pred'], 'argmax_pred'] = int(0)
train.loc[train['gmm_1'] == train['argmax_pred'], 'argmax_pred'] = int(1)
train.loc[train['gmm_2'] == train['argmax_pred'], 'argmax_pred'] = int(2)
train.loc[train['gmm_3'] == train['argmax_pred'], 'argmax_pred'] = int(3)
train.loc[train['gmm_4'] == train['argmax_pred'], 'argmax_pred'] = int(4)
train.loc[train['gmm_5'] == train['argmax_pred'], 'argmax_pred'] = int(5)
train.loc[train['gmm_6'] == train['argmax_pred'], 'argmax_pred'] = int(6)
train.loc[train['gmm_7'] == train['argmax_pred'], 'argmax_pred'] = int(7)
train.loc[train['gmm_8'] == train['argmax_pred'], 'argmax_pred'] = int(8)
train.loc[train['gmm_9'] == train['argmax_pred'], 'argmax_pred'] = int(9)
train.loc[train['gmm_10'] == train['argmax_pred'], 'argmax_pred'] = int(10)

test.loc[test['gmm_0'] == test['argmax_pred'], 'argmax_pred'] = int(0)
test.loc[test['gmm_1'] == test['argmax_pred'], 'argmax_pred'] = int(1)
test.loc[test['gmm_2'] == test['argmax_pred'], 'argmax_pred'] = int(2)
test.loc[test['gmm_3'] == test['argmax_pred'], 'argmax_pred'] = int(3)
test.loc[test['gmm_4'] == test['argmax_pred'], 'argmax_pred'] = int(4)
test.loc[test['gmm_5'] == test['argmax_pred'], 'argmax_pred'] = int(5)
test.loc[test['gmm_6'] == test['argmax_pred'], 'argmax_pred'] = int(6)
test.loc[test['gmm_7'] == test['argmax_pred'], 'argmax_pred'] = int(7)
test.loc[test['gmm_8'] == test['argmax_pred'], 'argmax_pred'] = int(8)
test.loc[test['gmm_9'] == test['argmax_pred'], 'argmax_pred'] = int(9)
test.loc[test['gmm_10'] == test['argmax_pred'], 'argmax_pred'] = int(10)

train['argmax_pred'] = train['argmax_pred'].astype(int)
test['argmax_pred'] = test['argmax_pred'].astype(int)

print(f1_score(train['open_channels'],train['prediction_round'], average = 'macro'))
print(f1_score(train['open_channels'],train['argmax_pred'], average = 'macro'))

train.to_csv('train_gmm_gold.csv',index=0)
test.to_csv('test_gmm_gold.csv',index=0)