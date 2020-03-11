import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
import pickle
from keras.utils import np_utils
from tensorflow.keras.layers import InputLayer, Dense, LSTM, Dropout, Activation
from tensorflow.keras.models import Sequential


fp = open('training_data.pkl', "rb")
training_data = pickle.load(fp)
fp.close()

ins_train=training_data['ins_train']
outs_train=training_data['outs_train']
ins_validation=training_data['ins_validation']
outs_validation=training_data['outs_validation']

n_vocab=244



model = Sequential()
model.add(LSTM(256,
               input_shape=(ins_train.shape[1], ins_train.shape[2]),
               return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['categorical_accuracy'])


# Callbacks
early_stopping_cb = keras.callbacks.EarlyStopping(patience=50,
                                                  restore_best_weights=True,
                                                  min_delta=0.01)

history=model.fit(ins_train, 
                  outs_train,
                  validation_data=(ins_validation, outs_validation),
                  epochs=500, 
                  batch_size=64,
                  callbacks=[early_stopping_cb],
                  use_multiprocessing=True)

model_name='Chopin_Prelude_Model1'


model.save_weights('./results/'+model_name+'_weights.h5')

results={}
results['history']=history.history


fp=open("{:s}_results.pkl".format(model_name),'wb')
pickle.dump(results,fp)
fp.close()

