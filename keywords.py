# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:12:52 2019

@author: Vishy
"""
from __future__ import absolute_import, division, print_function
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gensim.models as models
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


with open('lsa.txt', 'r') as myfile:
    data=myfile.read().replace('\n', '')



tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_lsa = tokenizer.tokenize(data)
    
def sentence_to_wordlist(raw):
    #raw = raw.lower()
   # ps = PorterStemmer()
    q=clean = re.sub("[^a-zA-Z]"," ", str(raw))
    q=q.lower()
    q=q.split()
    #text = [word for word in q if not word in set(stopwords.words('english'))]
    clean = re.sub("[^a-zA-Z]"," ", str(q))
    words = clean.split()
    return words

lsaclean = []
for i in raw_lsa:
    if len(i) > 0:
        lsaclean.append(str(sentence_to_wordlist(i)))

a=["a"]


lsaclean_set=set(lsaclean)

for i in lsaclean:
    lsaclean_set.add(i)


seen = set()
lsa_clean_dup = []
for item in lsaclean:
    if item not in seen:
        seen.add(item)
        lsa_clean_dup.append(item)



lsa_clean_list=[]
for i in lsa_clean_dup[0]:
    lsa_clean_list.append(i)

def sentence_to_wordlist(raw):
    #raw = raw.lower()
   # ps = PorterStemmer()
    q=clean = re.sub("[^a-zA-Z]"," ", str(raw))
    q=q.lower()
    q=q.split()
    #text = [word for word in q if not word in set(stopwords.words('english'))]
    clean = re.sub("[^a-zA-Z]"," ", str(q))
    words = clean.split()
    return words


for i in lsa_clean_dup: 
    if len(i) > 0:
        lsa_clean_list.append(sentence_to_wordlist(i))

with open('cleand.txt', 'w') as f:
    for item in lsa_clean_list:
        f.write(str(item))
    
