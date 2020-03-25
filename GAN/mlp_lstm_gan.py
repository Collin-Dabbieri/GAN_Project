
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
import pickle
from keras.utils import np_utils
from tensorflow.keras.layers import InputLayer, Dense, LSTM, Dropout, Activation, Bidirectional, Reshape, Input
from tensorflow.keras.models import Sequential, Model

from music21 import converter, instrument, note, chord, stream
import glob


FONTSIZE=18
plt.rcParams['figure.figsize']=(10,6)
plt.rcParams['font.size']=FONTSIZE

class GAN():
    
    def __init__(self):
        
        
        self.latent_dim=1000
        self.sequence_length=100
        self.sequence_shape=(self.sequence_length,1)
        
        self.lrate=0.001
        self.opt=tf.keras.optimizers.Adam(lr=lrate,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)