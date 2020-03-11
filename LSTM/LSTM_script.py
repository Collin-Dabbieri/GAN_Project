import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
import pickle
from keras.utils import np_utils
from tensorflow.keras.layers import InputLayer, Dense, LSTM, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse

def load_data(args):
    fp = open(args.data_file, "rb")
    training_data = pickle.load(fp)
    fp.close()
    
    ins_train=training_data['ins_train']
    outs_train=training_data['outs_train']
    ins_validation=training_data['ins_validation']
    outs_validation=training_data['outs_validation']
    
    args.num_notes=outs_train.shape[1]
    args.sequence_length=ins_train.shape[1]
    args.num_channels=ins_train.shape[2]
    
    return ins_train,outs_train,ins_validation,outs_validation


def build_model(args):
    
    num_LSTM_layers=len(args.LSTM)
    num_hidden_layers=len(args.hidden)

    model = Sequential()
    model.add(InputLayer(input_shape=(args.sequence_length,args.num_channels)))
    model.add(LSTM(args.LSTM[0],
                   activation=args.activation,
                   return_sequences=True,
                   kernel_regularizer=tf.keras.regularizers.l2(args.l2)))
    if args.dropout!=0:
        model.add(Dropout(args.dropout))
    
    
    for i in range(1,num_LSTM_layers-1):
        model.add(LSTM(args.LSTM[i],
                       activation=args.activation,
                       return_sequences=True,
                       kernel_regularizer=tf.keras.regularizers.l2(args.l2)))
        if args.dropout!=0:
            model.add(Dropout(args.dropout))
            
    model.add(LSTM(args.LSTM[-1],
                   activation=args.activation,
                   return_sequences=False,
                   kernel_regularizer=tf.keras.regularizers.l2(args.l2)))
        
    for i in range(num_hidden_layers):
        
        model.add(Dense(units=args.hidden[i],
                        activation=args.activation,
                        use_bias=True,
                        kernel_initializer=args.kernel_initializer,
                        bias_initializer=args.bias_initializer,
                        name='D'+str(i),
                        kernel_regularizer=tf.keras.regularizers.l2(args.l2)))
        
        if args.dropout!=0:
            model.add(Dropout(args.dropout))
            
            
    model.add(Dense(args.num_notes,
                    activation='softmax'))
    if args.verbose>0:
        print(model.summary())
    model.compile(loss=args.loss,optimizer='rmsprop',metrics=['categorical_accuracy'])
    
    return(model)

def generate_fname(args):
    
    if args.exp_index==-1:
        exp_str=''
        
    elif args.exp_index!=-1:
        exp_str='exp_'+str(args.exp_index)+'_'
    
    lstm_str='lstm_'
    for i in range(len(args.LSTM)):
        lstm_str+=str(args.LSTM[i])+'_'
        
    hidden_str='hidden_'
    for i in range(len(args.hidden)):
        hidden_str+=str(args.hidden[i])+'_'
        
    l2_str='l2_'+str(args.l2)+'_'
    dropout_str='drop_'+str(args.dropout)
    
    
    fbase=exp_str+lstm_str+hidden_str+l2_str+dropout_str
    
    if args.verbose>0:
        print('Filename Base is '+fbase)
        
    return fbase

def augment_args(args):
    '''
    The challenge here is that SLURM only gives one array for submitting multiple jobs
    So what we have to do is take one set of index values which can be from 0-999
    and transform that into all of the gridpoints we want
    '''
    #default value is -1 for which there is no change
    if args.exp_index==-1:
        return
    
    
    #lstm_options=[ [256,512,256],[256,256,256,256],[512,256,128] ]
    #hidden_options=[ [512,256],[256,128],[512,256,128] ]
    dropout_options=[ 0.2,0.3,0.4,0.5 ]
    l2_options=[ 0.01,0.001,0.0005,0.0001 ]
    num_gridpoints=len(dropout_options)*len(l2_options)
    
    assert(args.exp_index >= 0 and args.exp_index< num_gridpoints), "exp_index must be between 0 and {:.0f}".format(num_gridpoints)
    
    # so we have 16 total gridpoints here
    # the first 4 will have dropout=0.2
    # the next 4 will have dropout=0.3 and so on
    #so our SLURM array should cover 0-15
    
    idx_drop=int(args.exp_index/len(l2_options))
    idx_l2=args.exp_index%len(l2_options)
    
    args.dropout=dropout_options[idx_drop]
    args.l2=l2_options[idx_l2]
    
    
    
    
    
        

def execute_exp(args):
    
    augment_args(args)
    
    ins_train,outs_train,ins_validation,outs_validation=load_data(args)
    
    model=build_model(args)
    
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=0.01)
    
    fbase=generate_fname(args)
    filepath=args.results+fbase+"_checkpoint_weights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=args.verbose, save_best_only=True, mode='max')
    
    history=model.fit(ins_train, 
                  outs_train,
                  validation_data=(ins_validation, outs_validation),
                  use_multiprocessing=True,
                  epochs=args.epochs, 
                  batch_size=args.batch_size,
                  callbacks=[early_stopping_cb,checkpoint],
                  verbose=args.verbose)
    
    #Generate log data
    results={}
    results['args']=args
    results['predict_training']=model.predict(ins_train)
    results['predict_training_eval']=model.evaluate(ins_train,outs_train,verbose=args.verbose)
    results['predict_validation']=model.predict(ins_validation)
    results['predict_validation_eval']=model.evaluate(ins_validation,outs_validation,verbose=args.verbose)
    results['history']=history.history
    results['num_params']=model.count_params()
    
    
    #Save results
    results['fname_base']=fbase
    fp=open("{:s}{:s}_results.pkl".format(args.results,fbase),'wb')
    pickle.dump(results,fp)
    fp.close()
    
    # Model
    model.save("{:s}{:s}_model".format(args.results,fbase))
    return model
    
def create_parser():
    
    parser=argparse.ArgumentParser(description='Music Learner')
    
    parser.add_argument('-data_file',type=str,default='./training_data.pkl',help='path to training pickle file')
    parser.add_argument('-LSTM',nargs='+',type=int,default=[256,512,256],help='number of LSTM nodes per layer')
    parser.add_argument('-hidden',nargs='+',type=int,default=[512,256],help='number of hidden nodes per layer')
    parser.add_argument('-activation',type=str,default='elu',help='activation of LSTM and hidden nodes')
    parser.add_argument('-l2',type=float,default=0.001,help='l2 regularizer value')
    parser.add_argument('-dropout',type=float,default=0.3,help='dropout probability')
    parser.add_argument('-kernel_initializer',type=str,default='truncated_normal',help='kernel initializer for dense layers')
    parser.add_argument('-bias_initializer',type=str,default='zeros',help='bias initializer for dense layers')
    parser.add_argument('-loss',type=str,default='categorical_crossentropy',help='loss function for training')
    parser.add_argument('-patience',type=int,default=100,help='patience for early stopping callback')
    parser.add_argument('-epochs',type=int,default=500,help='number of training epochs')
    parser.add_argument('-batch_size',type=int,default=64,help='training batch size')
    parser.add_argument('-exp_index',type=int,default=-1,help='experiment index for augment args')
    parser.add_argument('-results',type=str,default='./results/',help='directory for storing results')
    parser.add_argument('-verbose',type=int,default=1,help='verbosity')
    return parser


#################################################################
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    execute_exp(args)
    

    
    