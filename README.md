# Identifying Tweets against the flu shot using deep learning models

**December 2017**

This repository contains the code I implemented for a project that needed to identify in real time tweets that were against the flu shot. The system used deep learning models, like CNN and LSTM with attention and embeddings for natural language processing.   

The code was implemented on December 2017, during my graduate studies at the Master of Information and Data Science (MIDS) 
program at UC Berkeley. The class was: W241 Experiments and Causal Inference 

## Labeling Tweets that mention 'flu shot'   
We has tweets collected by the search term 'flu shot', the next step was to label them with "Y" or "N" to indicate if they are speaking against the flue shot or not. I worked with MTurk for labeling our dataset.
The Labeling process was done in different steps

| Step | Task | Program |
|---|---|----|
|1 | Change format of the file to 5 tweets in a row in order for MTurkers to label 5 tweets at a time | [labeling_tweets/src/Converting_To_Mturk_Format.ipynb](labeling_tweets/src/Converting_To_Mturk_Format.ipynb) |
|2| Mturk custom template to handle 5 tweets in the same review, it is faster for mturkers. Each record was reviewed by 3 mturkers|[labeling_tweets/src/mturk_flu_tweet_custom_template.txt](labeling_tweets/src/mturk_flu_tweet_custom_template.txt) |
|3| Upload files to Mturk with custom template.. wait for Mturkers to label.. download results | |
|4| Mturk results were files with 5 tweets in each row, repeated 3 times with the labels from the 3 mturkers. This program produce a file with one row for each tweet and consolidating mturkers label by choosing the label with more votes |[labeling_tweets/src/processing_mturk_file_3workers.ipynb](labeling_tweets/src/processing_mturk_file_3workers.ipynb) |

Mturkers labeled 7000 tweets.

## Training Model to Classify Tweets

Training a model to classify tweets that are speaking against the flu shot 

| Model | Notebook | roc_auc |
|----|----|---|
|Baseline Logistic Regression | [classifiers/twitter_flue_baseline.ipynb](classifiers/twitter_flue_baseline.ipynb)|83%|
|CNN with Embeddings | [classifiers/CNN_LSTM_classifier.ipynb](classifiers/CNN_LSTM_classifier.ipynb)|88%|
|LSTM with an Attention layer and using Embeddings | [classifiers/CNN_LSTM_classifier.ipynb](classifiers/CNN_LSTM_classifier.ipynb)|90%|

## Model Python modules

| Model | Python module |
| --- | --- |
| CNN | [classifiers/icd9_cnn_model.py](classifiers/icd9_cnn_model.py)  |
| LSTM_ATT | [classifiers/icd9_lstm_att_model.py](classifiers/icd9_lstm_att_model.py)   |
| Attention Layer |[classifiers/attention_util.py](classifiers/attention_util.py)  |



