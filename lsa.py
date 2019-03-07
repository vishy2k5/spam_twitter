# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:07:45 2019

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




dataset_tweets=pd.read_csv('tweet.csv')

tweets=pd.read_csv('spam_tweets.tsv', delimiter='\t')

tweets.columns = ['id','tid','tweet','date']

tweets_red=tweets[:300000]


twitter=tweets_red['tweet'].tolist()


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_tweet = tokenizer.tokenize(str(twitter))


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def sentence_to_wordlist(raw):
    #raw = raw.lower()
   # ps = PorterStemmer()
    q=clean = re.sub("[^a-zA-Z]"," ", str(raw))
    q=q.lower()
    q=q.split()
    text = [word for word in q if not word in set(stopwords.words('english'))]
    clean = re.sub("[^a-zA-Z]"," ", str(text))
    words = clean.split()
    return words

sentenceslsa = []
for i in raw_tweet:
    if len(i) > 0:
        sentenceslsa.append(sentence_to_wordlist(i))
        

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

dct1 = Dictionary(sentenceslsa) 

corpus_lsa = [dct1.doc2bow(line) for line in sentenceslsa] 
model = TfidfModel(corpus_lsa)
corpus_tfidf1 = model[corpus_lsa]

for doc in corpus_tfidf1:
    print(doc)
    
lsi = models.LsiModel(corpus_tfidf1, id2word=dct1, num_topics=20)

corpus_lsi = lsi[corpus_tfidf1]

lsi.print_topic(9)

lsi.print_topics(20)

lsi.show_topic()

top=lsi.show_topics()

lsi.save('a')

print(top)

top.


with open('lsa.txt', 'w') as f:
    for item in top:
        f.write(str(item))
















num_features = 300
# Minimum wohe RNG, to make the results reproducible.
#random number generator
#determrd count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for tinistic, good for debugging
seed = 1

data2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


data2vec.build_vocab(sentenceslsa)



data2vec.train(sentenceslsa, total_examples=data2vec.corpus_count, epochs=15)

data2vec.most_similar('kevinmax')

data2vec.save('www')

lexical_diversity(sentenceslsa)

fdist=FreqDist(sentenceslsa)

data2vec.wv.save_word2vec_format(fname="vectors.txt", fvocab=None, binary=False)






book_filenames = sorted(glob.glob("lsa.txt"))



f = open('lsa.txt', 'r')














