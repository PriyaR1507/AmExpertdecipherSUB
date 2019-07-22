import pandas as pd
import numpy as np
import keras
from keras.layers import Dense,LSTM,GRU,BatchNormalization,Dropout,concatenate,Bidirectional,Input,TimeDistributed,\
                        Conv1D,Flatten,AveragePooling1D,MaxPooling1D
from keras.optimizers import Adam,RMSprop
from keras.models import Model
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle

f = open('data2','rb')
data = pickle.load(f)

def preprocess(df):
    for col in df.columns:
        df[col] /=max(df[col])
    return df
data = preprocess(data)
tr = data[:32820]

y_te = pd.read_csv('test_9K3DBWQ.csv')
y = pd.read_csv('train_bFQbE3f/train.csv')['cc_cons']

def preprocess_y(y_):
    return y_/max(y_)
y = preprocess_y(y)
max_y = max(y)

i = Input(shape=(79,1))
x = LSTM(50,return_sequences=True,recurrent_dropout=0.1)(i)
x = BatchNormalization()(x)
x = Conv1D(20,3,activation='relu')(x)
x1 = MaxPooling1D()(x)
x2 = AveragePooling1D()(x)
x = concatenate([x1,x2])
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(50,activation='relu')(x)
x = Dense(1)(x)

model = Model(i,x)
model.compile(optimizer='adam',loss='mse',metrics=['mse'])

from keras.callbacks import ModelCheckpoint
filepath = 'model2-{epoch:02d}-{val_mean_squared_error:.5f}.hdf5'
chk = ModelCheckpoint(filepath=filepath,monitor='val_mean_squared_error', verbose=1, save_best_only=False, mode='min')
history = model.fit(np.expand_dims(tr,-1),y.values,epochs=35,batch_size=128,
			validation_split=0.15,callbacks=[chk])
with open('history','rb') as f:
    pickle.dump(history,f)
 
