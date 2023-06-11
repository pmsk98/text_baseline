# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 19:02:42 2023

@author: user
"""

import pandas as pd
import numpy as np


train = pd.read_csv('C:/Users/user/Desktop/dacon_법원 판결/train.csv')

test = pd.read_csv('C:/Users/user/Desktop/dacon_법원 판결/test.csv')

sample_sub = pd.read_csv('C:/Users/user/Desktop/dacon_법원 판결/sample_submission.csv')


#%%
train.columns


# import torch
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

#%%
import ktrain
from ktrain import text

train.columns
train_text = train['first_party'].values.tolist()

train_output =train['first_party_winner'].values.tolist()

classname=[0,1]
classname=np.array(classname)

# model_name="distilbert-base-uncased-finetuned-sst-2-english"
classifier= text.Transformer(model_name,maxlen=512,class_names=classname)



train_data=classifier.preprocess_train(train_text,train_output)


model=classifier.get_classifier()


learner= ktrain.get_learner(model,train_data=train_data,batch_size=8)


learner.fit_onecycle(1e-5,5)

#%%

test_text = test['facts'].values.tolist()

predictor= ktrain.get_predictor(learner.model,preproc=classifier)


a = predictor.predict(test_text)


a = pd.DataFrame(a)

a.value_counts()


sample_sub['first_party_winner'] = a[0]

sample_sub['first_party_winner'].value_counts()


sample_sub.to_csv('C:/Users/user/Desktop/dacon_법원 판결/distilbert_first_party.csv',index=False)

