!pip install tensorflow_addons==0.9.1 
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, Dense, Add, Multiply,SeparableConv1D,UpSampling1D
import pandas as pd
import numpy as np
import random
from tensorflow.keras.callbacks import Callback, LearningRateScheduler,EarlyStopping
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, Nadam, SGD, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models, optimizers
import tensorflow_addons as tfa
import gc
from tensorflow.keras.layers import BatchNormalization, SpatialDropout1D,AveragePooling1D
#from keras_contrib.layers import CRF


from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import os

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

# configurations and main hyperparammeters
EPOCHS = 150
NNBATCHSIZE = 16
GROUP_BATCH_SIZE = 4000
SEED = 321
LR = 0.001
SPLITS = 5

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

# read data
def read_data():
    train = pd.read_csv('../input/50hz-noise-2-724/train_kalman_clean.csv', usecols = ['time','clear_signal','open_channels']) #3 v
    test  = pd.read_csv('../input/50hz-noise-2-724//test_kalman_clean.csv',usecols=['time','clear_signal'])
    sub  = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})
    
    train.rename(columns = {'clear_signal':'signal'},inplace=True)
    test.rename(columns = {'clear_signal':'signal'},inplace=True)
    
    train_gmm = pd.read_csv('../input/gold-gmm/train_gmm_gold')
    test_gmm = pd.read_csv('../input/gold-gmm/test_gmm_gold')
    
    train = pd.concat([train,train_gmm[['prediction_float','gmm_0','gmm_1','gmm_2',
                                       'gmm_3','gmm_4','gmm_5','gmm_6','gmm_7','gmm_8',
                                       'gmm_9','gmm_10']]],axis=1)
    
    test = pd.concat([test,test_gmm[['prediction_float','gmm_0','gmm_1','gmm_2',
                                       'gmm_3','gmm_4','gmm_5','gmm_6','gmm_7','gmm_8',
                                       'gmm_9','gmm_10']]],axis=1)
    
    del train_gmm, test_gmm
    gc.collect()
    
    train.rename(columns = {'prediction_float':'prediction'},inplace=True)
    test.rename(columns = {'prediction_float':'prediction'},inplace=True)
    
    return train, test, sub

# create batches of 4000 observations
def batching(df, batch_size):
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# normalize the data (standard scaler). We can also try other scalers for a better score!
def normalize(train, test):
    
    sn = pd.concat([train,test],axis=0)
    mu = sn['signal'].mean()
    sigma = sn['signal'].std()

    train['signal'] = (train.signal - mu) / sigma
    test['signal'] = (test.signal - mu) / sigma
    
    del sn
    gc.collect()

    return train, test

# get lead and lags features
def lag_with_pct_change(df, windows):
    for window in windows:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df

def sq_lag_with_pct_change(df, windows):
    for window in windows:
        df['sq_signal_shift_pos_' + str(window)] = df.groupby('group')['signal_2'].shift(window).fillna(0)
        df['sq_signal_shift_neg_' + str(window)] = df.groupby('group')['signal_2'].shift(-1*window).fillna(0)
        return df

def diff_with_pct_change(df,windows):
    for window in windows:
        df['signal_diff_pos_' + str(window)] = df.groupby('group')['signal'].diff(window).fillna(0)
        df['signal_diff_neg_' + str(window)] = df.groupby('group')['signal'].diff(-1 * window).fillna(0)
    return df



def lag_with_pct_change_gmm(df, windows):
    for window in windows:    
        df['prediction_shift_pos_' + str(window)] = df.groupby('group')['prediction'].shift(window).fillna(0)
        df['prediction_shift_neg_' + str(window)] = df.groupby('group')['prediction'].shift(-1 * window).fillna(0)
    return df

def diff_with_pct_change_gmm(df,windows):
    for window in windows:
        df['prediction_diff_pos_' + str(window)] = np.abs(df.groupby('group')['prediction'].diff(window).fillna(0))
        df['prediction_diff_neg_' + str(window)] = np.abs(df.groupby('group')['prediction'].diff(-1 * window).fillna(0))
    return df


# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, batch_size):
    # create batches
    df = batching(df, batch_size = batch_size)
    # create leads and lags (1, 2, 3 making them 6 features)
    df = lag_with_pct_change(df, [1,2])
    ###df = diff_with_pct_change(df, [1, 2, 3])
    # create signal ** 2 (this is the new feature)
    df['signal_2'] = df['signal'] ** 2
    ###df = sq_lag_with_pct_change(df, [1, 2, 3]) 
    ###df['sin_signal'] = np.sin(df['signal'])
    ###df['cos_signal'] = np.cos(df['signal'])
    ###df['tan_signal'] = np.tan(df['signal'])
    ######df['signal_3'] = df['signal'] ** 3
    #df['prediction_2'] = df['prediction'] ** 2

    df = lag_with_pct_change_gmm(df, [1])
    #df = diff_with_pct_change_gmm(df, [1])
    #df = lag_with_pct_change_gmm2(df, [1])
    #df['root_signal'] = np.sqrt(np.abs(df['signal']))
    #df = rolling_stats(df, [10, 50, 100, 500,  1000])


    return df

# fillna with the mean and select features for training
def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features

# model function (very important, you can try different arquitectures to get a better score. I believe that top public leaderboard is a 1D Conv + RNN style)
def Classifier(shape_):
    
    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)

            #x = BatchNormalization()(x)
            #x = SpatialDropout1D(0.2)(x)
                    
            res_x = Add()([res_x, x])
        return res_x
    
    inp = Input(shape = (shape_))
    
    x = wave_block(inp, 16, 3, 12)
    x = wave_block(x, 16, 3, 12)
    x = wave_block(x, 32, 3, 8)
    x = wave_block(x, 64, 3, 4)
    #x = AveragePooling1D(10)(x)
    x = wave_block(x, 128, 3, 1)
    #x = Bidirectional(LSTM(32, return_sequences=True))(x)
    #x = Dropout(0.2)(x)
    #x = Bidirectional(LSTM(16, return_sequences=True))(x)
    #x = Attention(150)(x)
    #x = Dropout(0.2)(x)
    #x = TimeDistributed(Dense(32, activation = 'relu'))(x)
    out = Dense(11, activation = 'softmax', name = 'out')(x)
    
    model = models.Model(inputs = inp, outputs = out)
    
    opt = Adam(lr = LR)
    opt = tfa.optimizers.SWA(opt)
    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])
    return model

# function that decrease the learning as epochs increase (i also change this part of the code)
def lr_schedule(epoch):
    if epoch < 30:
        lr = LR
    elif epoch < 40:
        lr = LR / 3
    elif epoch < 50:
        lr = LR / 5
    elif epoch < 60:
        lr = LR / 7
    elif epoch < 70:
        lr = LR / 9
    elif epoch < 80:
        lr = LR / 11
    elif epoch < 90:
        lr = LR / 13
    elif epoch < 110:
        lr = LR / 15
    elif epoch < 130:
        lr = LR / 100
    else:
        lr = LR / 200
    return lr

# class to get macro f1 score. This is not entirely necessary but it's fun to check f1 score of each epoch (be carefull, if you use this function early stopping callback will not work)
class MacroF1(Callback):
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis = 2).reshape(-1)
        
    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis = 2).reshape(-1)
        score = f1_score(self.targets, pred, average = 'macro')
        print(f'F1 Macro Score: {score:.5f}')

# main function to perfrom groupkfold cross validation (we have 1000 vectores of 4000 rows and 8 features (columns)). Going to make 5 groups with this subgroups.
def run_cv_model_by_batch(train, test, splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size):
    
    seed_everything(SEED)
    K.clear_session()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    oof_ = np.zeros((len(train), 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)
    preds_ = np.zeros((len(test), 11))
    target = ['open_channels']
    group = train['group']
    kf = GroupKFold(n_splits=5)
    splits = [x for x in kf.split(train, train[target], group)]
    oof_pd = pd.DataFrame()
    oof_pd['open_channels'] = train['open_channels']

    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])    
        new_splits.append(new_split)
    # pivot target columns to transform the net to a multiclass classification estructure (you can also leave it in 1 vector with sparsecategoricalcrossentropy loss function)
    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)

    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']
    target_cols = ['target_'+str(i) for i in range(11)]
    
    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))

    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):
        train_x, train_y = train[tr_idx], train_tr[tr_idx]
        valid_x, valid_y = train[val_idx], train_tr[val_idx]
        print(f'Our training dataset shape is {train_x.shape}')
        print(f'Our validation dataset shape is {valid_x.shape}')

        train_x_new = train

        gc.collect()
        shape_ = (None, train_x.shape[2]) # input is going to be the number of feature we are using (dimension 2 of 0, 1, 2)
        model = Classifier(shape_)
        # using our lr_schedule function
        cb_lr_schedule = LearningRateScheduler(lr_schedule)
        # Use Early-Stopping
        callback_early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        model.fit(train_x,train_y,
                  epochs = nn_epochs,
                  callbacks = [callback_early_stopping,cb_lr_schedule, MacroF1(model, valid_x, valid_y)], # adding custom evaluation metric for each epoch
                  batch_size = nn_batch_size,verbose = 2,
                  validation_data = (valid_x,valid_y))
        preds_f = model.predict(valid_x)
        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro') # need to get the class with the biggest probability
        print(f'Training fold {n_fold + 1} completed. macro f1 score : {f1_score_ :1.5f}')
        preds_f = preds_f.reshape(-1, preds_f.shape[-1])
        oof_[val_orig_idx,:] += preds_f
        te_preds = model.predict(test)
        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           
        preds_ += te_preds / SPLITS
        del train_x, train_y, valid_x, valid_y
    # calculate the oof macro f1_score
    f1_score_ = f1_score(np.argmax(train_tr, axis = 2).reshape(-1),  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)
    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')
    sample_submission['open_channels'] = np.argmax(preds_, axis = 1).astype(int)
    oof_pd['open_channels_pred'] = np.argmax(oof_, axis = 1).astype(int)
    sample_submission.to_csv('submission_wavenet.csv', index=False, float_format='%.4f')
    #oof_pd.to_csv('oof_wavenet.csv', index=False, float_format='%.4f')
    #oof_proba = pd.DataFrame(oof_)
    #sub_proba = pd.DataFrame(preds_)
    #oof_proba.to_csv('oof_proba.csv',index=0)
    #sub_proba.to_csv('sub_proba.csv',index=0)
    
# this function run our entire program
def run_everything():
    
    print('Reading Data Started...')
    train, test, sample_submission = read_data()
    train, test = normalize(train, test)
    print('Reading and Normalizing Data Completed')
        
    print('Creating Features')
    print('Feature Engineering Started...')
    #train, test = gmm_feature(train,test)
    train = run_feat_engineering(train, batch_size = GROUP_BATCH_SIZE)
    test = run_feat_engineering(test, batch_size = GROUP_BATCH_SIZE)
    train, test, features = feature_selection(train, test)
    print('Feature Engineering Completed...')
   
    print(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started...')
    run_cv_model_by_batch(train, test, SPLITS, 'group', features, sample_submission, EPOCHS, NNBATCHSIZE)
    print('Training completed...')
        
run_everything()