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
|2| Mturk custom template to handle 5 tweets in the same review, it is faster for mturkers. Each record was reviewed by 3 mturkers|[labeling_tweets/src/mturk flu tweet custom template.txt](labeling_tweets/src/mturk flu tweet custom template.txt) |

