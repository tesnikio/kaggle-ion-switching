import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import os.path as path
import numpy.fft as fft
from scipy import signal
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set(rc={'figure.figsize':(22,10)})

dataset_path = path.join("..", "input", "detrendedwithkalman")
train = read_csv(path.join(dataset_path, "train_kalman.csv"))
test = read_csv(path.join(dataset_path, "test_kalman.csv"))


train.loc[(train.index//500000 == 4) | (train.index//500000 == 9), 'signal'] += 2.724
test.loc[(test.index//100000 == 5) | (test.index//100000 == 7), 'signal'] += 2.724 + 0.04


train = train[train.index // 500000 != 7].reset_index(drop=True)

dataset_path = path.join("..", "input", "default-old-shift")
pseudo = read_csv(path.join(dataset_path, "submission_wavenet.csv"))
test['open_channels'] = pseudo['open_channels']

data = train.append(test).reset_index(drop=True)

c = 11   # channel size
label = np.arange(len(train['signal']))

channel_list = np.arange(c)
n_list = np.empty(c)
mean_list = np.empty(c)
std_list = np.empty(c)
stderr_list = np.empty(c)

for i in range(c):
    x = label[train['open_channels'] == i]
    y = train['signal'][train['open_channels'] == i]
    n_list[i] = np.size(y)
    mean_list[i] = np.mean(y)
    std_list[i] = np.std(y)    
stderr_list = std_list / np.sqrt(n_list)


# Predict general mean
w = 1 / stderr_list
channel_list = channel_list.reshape(-1, 1)
linreg_m = LinearRegression()
linreg_m.fit(channel_list, mean_list, sample_weight = w)

mean_predict = linreg_m.predict(channel_list)

x = np.linspace(-0.5, 5.5, c)
y = linreg_m.predict(x.reshape(-1, 1))
plt.plot(x, y, label = "regression")
plt.plot(channel_list, mean_list, ".", markersize = 8, label = "original")
plt.legend()
plt.show()

print("mean_predict :", mean_predict)

def Arrange_mean(signal, channels, diff, channel_range, reverse=False):
    signal_out = signal.copy()
    if reverse:
        diff = -diff
    
    for i in range(channel_range):
        signal_out[channels == i] -= diff[i]
    return signal_out

def bandstop(x, samplerate = 1000000, fp = np.array([4960, 5040]), fs = np.array([4930, 5070])):
    fn = samplerate / 2  # Nyquist frequency
    wp = fp / fn
    ws = fs / fn
    gpass = 1
    gstop = 3

    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "bandstop")
    y = signal.filtfilt(b, a, x)
    return y

def highpass(x, samplerate = 1000000, fp = 30, fs = 50):
    fn = samplerate / 2  # Nyquist frequency
    wp = fp / fn
    ws = fs / fn
    gpass = 1
    gstop = 3

    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "highpass")
    y = signal.filtfilt(b, a, x)
    return y


data['signal_without_channel'] = Arrange_mean(data['signal'], data['open_channels'], mean_predict, 11)

plt.figure(figsize = (16, 10))
f, t, Zxx = signal.stft(data['signal_without_channel'], 1/0.0001, nperseg=10000)
plt.pcolormesh(t, f, np.abs(Zxx))
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('log')
plt.show()

# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.ylim(top=60)
# plt.ylim(bottom=40)
# plt.show()

data['clear_signal'] = bandstop(data['signal_without_channel'])
plt.figure(figsize = (16, 10))
f, t, Zxx = signal.stft(data['clear_signal'], 1/0.0001, nperseg=10000)
plt.pcolormesh(t, f, np.abs(Zxx))
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('log')
plt.show()

# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.ylim(top=60)
# plt.ylim(bottom=40)
# plt.show()

data.loc[~(((data.index>=500000*2) & (data.index < 500000*3)) |
         ((data.index>=500000*7) & (data.index < 500000*7)) |
         ((data.index>=4500000 + 100000*4) & (data.index < 4500000 + 100000*5))), 'clear_signal'] =\
    highpass(data.loc[~(((data.index>=500000*2) & (data.index < 500000*3)) |
         ((data.index>=500000*7) & (data.index < 500000*7)) |
         ((data.index>=4500000 + 100000*4) & (data.index < 4500000 + 100000*5))), 'clear_signal'])
plt.figure(figsize = (16, 10))
f, t, Zxx = signal.stft(data['clear_signal'], 1/0.0001, nperseg=10000)
plt.pcolormesh(t, f, np.abs(Zxx))
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('log')
plt.show()

# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.ylim(top=5)
# plt.ylim(bottom=0)
# plt.show()

data['clear_signal'] = Arrange_mean(data['clear_signal'], data['open_channels'], mean_predict, 11, True)

train = data[:train.shape[0]].reset_index(drop=True)
test = data[train.shape[0]:].reset_index(drop=True)

train[['time', 'signal', 'clear_signal', 'open_channels']].to_csv('train_kalman_clean.csv', index=False)
test[['time', 'signal', 'clear_signal']].to_csv('test_kalman_clean.csv', index=False)

import seaborn as sns
sns.set(rc={'figure.figsize':(22,10)})
sns.lineplot(data['time'].head(8000), data['signal'].head(8000)-data['clear_signal'].head(8000))
# sns.lineplot(data['time'].head(8000), data['signal'].head(8000))
# sns.lineplot(data['time'].head(8000), data['clear_signal'].head(8000))

sns.distplot(data['signal'], hist=False)
sns.distplot(data['clear_signal'], hist=False)

def batching(df, batch_size):
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df
train = batching(train,4000)
# train_6 = train[(train.index//500000 != 4) & (train.index//500000 != 9)].copy()
# train_11 = train[(train.index//500000 == 4) | (train.index//500000 == 9)].copy()
trains = [train]

from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
feats = ['signal', 'clear_signal']
models = {
#     'QDA': QuadraticDiscriminantAnalysis(),
#     'LDA': LinearDiscriminantAnalysis(),
    'GNB': GaussianNB()
}


kf = GroupKFold(n_splits = 5)

for idxs in zip(*tuple(kf.split(trains[i]['signal'], trains[i]['open_channels'], trains[i]['group']) for i in range(len(trains)))):
    print('='*130)
    for key in models.keys():
        print(key, end=': ' )
        for feat in feats:
            
            y_pred = []
            y_test = []
            for i in range(len(trains)):
                train_index, test_index = idxs[i]

                X = trains[i][feat].values.reshape(-1, 1)
                y = trains[i]['open_channels'].values
                
                X_train, X_valid = X[train_index], X[test_index]
                y_train, y_valid = y[train_index], y[test_index]
                

                models[key].fit(X_train,y_train)
                y_pred.append(models[key].predict(X_valid))
                y_test.append(y_valid)
                
            print(feat, round(f1_score(np.hstack(y_test), np.hstack(y_pred), average = 'macro'), 6), end='; ')
        print()