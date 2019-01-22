import re
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.metrics import confusion_matrix

#------------------------------------------------------------------------------------
#Preprocessing Tweets
#------------------------------------------------------------------------------------
def pre_process_tweets(data):
    return map(lambda x: process_str(x),data)

def process_str(string):

    string = string.lower()
    string = string.replace("\n", " ") # remove the lines
    string = re.sub("[^a-zA-Z0-9\ \']+", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    """ Canonize numbers"""
    string = re.sub(r"(\d+)", "DG", string)
    
    return string.strip()

def vectorize_tweets(col, MAX_NB_WORDS, verbose = True):
    """Takes a note column and encodes it into a series of integer
        Also returns the dictionnary mapping the word to the integer"""
    tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
    tokenizer.fit_on_texts(col)
    data = tokenizer.texts_to_sequences(col)
    note_length =  [len(x) for x in data]
    vocab = tokenizer.word_index
    MAX_VOCAB = len(vocab)
    if verbose:
        print('Vocabulary size: %s' % MAX_VOCAB)
        print('Average note length: %s' % np.mean(note_length))
        print('Max note length: %s' % np.max(note_length))
    return data, vocab, MAX_VOCAB, tokenizer 

def pad_tweets(data, MAX_SEQ_LENGTH):
    data = pad_sequences(data, maxlen = MAX_SEQ_LENGTH)
    return data, data.shape[1]
#------------------------------------------------------------------------------------
# Vectorize labels
#------------------------------------------------------------------------------------

def vectorize_label (labels):
    vlabels=[1 if element == "Y" else 0 for element in labels]
    return vlabels

#------------------------------------------------------------------------------------
# Split data into train, dev, test
#------------------------------------------------------------------------------------
def split_data(data, train_frac = 0.7, dev_frac = 0.15):   
    train_split_idx = int(train_frac * len(data))
    dev_split_idx = int ((train_frac + dev_frac)* len(data))
    train_data = data[:train_split_idx]
    dev_data = data[train_split_idx:dev_split_idx]
    test_data = data[dev_split_idx:]
    return train_data, dev_data, test_data

def split_data2(data, train_frac = 0.8):   
    train_split_idx = int(train_frac * len(data))
    train_data = data[:train_split_idx]
    dev_data = data[train_split_idx:]
    return train_data, dev_data
#------------------------------------------------------------------------------------
# For Prediction
#------------------------------------------------------------------------------------
def preprocess_tweets(tweets, tokenizer,MAX_SEQ_LENGTH):
    tweets_text = pre_process_tweets (tweets)
    data_vectorized = tokenizer.texts_to_sequences(tweets)    
    tweets_data, MAX_SEQ_LENGTH = pad_tweets(data_vectorized, MAX_SEQ_LENGTH)
    return tweets_data

#------------------------------------------------------------------------------------
# F1-score
#------------------------------------------------------------------------------------
def get_f1_score(y_true,y_hat,threshold, average):
    hot_y = np.where(np.array(y_hat) > threshold, 1, 0)
    return f1_score(np.array(y_true), hot_y, average=average)

def show_f1_score(y_train, pred_train, y_val, pred_dev):
    print('F1 scores')
    print('threshold | training | dev  ')
    f1_score_average = 'micro'
    for threshold in [  0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7,  0.8]:
        train_f1 = get_f1_score(y_train, pred_train,threshold,f1_score_average)
        dev_f1 = get_f1_score(y_val, pred_dev,threshold,f1_score_average)
        print('%1.3f:      %1.3f      %1.3f' % (threshold,train_f1, dev_f1))

#------------------------------------------------------------------------------------
# scores
#------------------------------------------------------------------------------------
def show_scores( y_true, pred_probabilities):
    print('F1 scores')
    print('threshold   |  tn   |  fp  |   fn   |  tp   | f1 score                  | recall  |precision|')
    print '-' * 95 
    for threshold in [  0.05, 0.1,0.15, 0.2,0.25, 0.3, 0.4, 0.5, 0.6, 0.7,  0.8, 0.9, 0.95]:
        pred_label = np.where(np.array(pred_probabilities) > threshold, 1, 0)
        f1a, f1b =f1_score(y_true, pred_label, average=None)
        f1 =f1_score(y_true, pred_label, average='micro')
        recall = recall_score(y_true, pred_label)
        precision = precision_score(y_true, pred_label)
        tn, fp, fn, tp = confusion_matrix(y_true, pred_label).ravel()       
        print('%1.3f:      %5d   %5d   %5d   %5d    ( %1.3f ,  %1.3f ) = %1.3f   %1.3f     %1.3f  ' % 
              (threshold, tn.item(), fp.item(), fn.item(), tp.item(), f1a, f1b,f1, recall, precision))
        print '-' * 95