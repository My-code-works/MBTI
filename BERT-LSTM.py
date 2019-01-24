from keras.models import Sequential, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers.crf import CRF
from keras_contrib.utils import save_load_utils
from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import pickle, os
from utils import *
from Dataprocessor import Dataprocessor
from bert_utils import get_all_features


BERT_BASE = os.path.join(os.getcwd(), 'bert/bert_model/uncased_L-12_H-768_A-12')
INPUT_LENGTH = 100
PARAGRAPH_EMB_DIM = 768
NUM_TAGS = 12
bert_config_file = os.path.join(BERT_BASE, 'bert_config.json')
vocab_file = os.path.join(BERT_BASE, 'vocab.txt')
bert_checkpoint = os.path.join(BERT_BASE, 'bert_model.ckpt')


class LSTMmodel:
    def __init__(self, input_length, para_emb_dim, num_tags, hidden_dim=200, dropout=0.5):
        self.num_tags = num_tags
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True), input_shape=(input_length, para_emb_dim)))
        self.model.add(Dropout(dropout))
        # self.model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True), input_shape=(input_length, para_emb_dim)))
        # self.model.add(Dropout(dropout))
        self.model.add(TimeDistributed(Dense(self.num_tags)))
        crf = CRF(self.num_tags)
        self.model.add(crf)
        self.model.compile('rmsprop', loss=crf_loss, metrics=[crf_accuracy])
    
    def save_model(self, filepath):
        save_load_utils.save_all_weights(self.model, filepath)
    
    def restore_model(self, filepath):
        save_load_utils.load_all_weights(self.model, filepath)
        
    def train(self, trainX, trainY, batch_size=32, epochs=10, validation_split=0.1, verbose=1):
        return self.model.fit(trainX, np.array(trainY), batch_size=batch_size, epochs=epochs, 
                             validation_split=validation_split, verbose=verbose)
    
    @staticmethod
    def myloss(y_true, y_pred):   
        y_pred /= tf.reduce_sum(y_pred, -1, True)
        # manual computation of crossentropy
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        return -tf.reduce_sum(y_true * tf.log(y_pred), -1)
      

class InferenceModel:
    def __init__(self, model_weight_path, input_length=INPUT_LENGTH, para_dim=PARAGRAPH_EMB_DIM, num_tags=NUM_TAGS):
        self.LSTMmodel = LSTMmodel(input_length, para_dim, num_tags)
        self.input_length = input_length
        self.para_dim = para_dim
        self.num_tags = num_tags
        self.dataprocessor = Dataprocessor()
        self.LSTMmodel.model.load_weights(model_weight_path)
    
    def infer(self, parag_list):
        feature = get_all_features([parag_list], bert_config_file, vocab_file, bert_checkpoint)
        
        X = [] # X is 3D: article, paragraph, embedding; Y is 2D: article, paragraph
        for f in feature:
            while len(f) < self.input_length:
                f.append(np.zeros(self.para_dim))
            f = f[0:self.input_length]
            X.append(f)

        test_pred = self.LSTMmodel.model.predict(np.array(testX), verbose=1)
        labels = []
        test_pred = test_pred[0]
        for p in test_pred:
            labels.append(self.dataprocessor.gettag(np.argmax(p)))
        return labels

            
if __name__ == '__main__':
    model_weight_path = 'save_model/base_100_3_6.h5'
    infermodel = InferenceModel(model_weight_path)
    
    article = ['Some', 'Paragraphs', 'To', 'Tag']
    labels = infermodel.infer(article)
