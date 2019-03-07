# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:30:47 2019

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



dataset_tweets=pd.read_csv('spam_tweets.tsv', delimiter='\t')

columns = dataset_tweets.columns


dataset_tweets.columns = ['id','tid','tweet','date']

df1=dataset_tweets[['tweet']]

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')




#raw_sentences = tokenizer.tokenize(df)
raw=''

import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 2333690):
    review = re.sub('[^a-zA-Z]', ' ', dataset_tweets['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

print(str1)

str1 = ''.join(corpus)

raw_sentences = tokenizer.tokenize(str1)

raw_sentences12 = tokenizer.tokenize(str(df1))



def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

token_count = sum([len(sentence) for sentence in sentences])
print("The data contains {0:,} tokens".format(token_count))



#more dimensions = more generalized
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

data2vec.build_vocab(sentences)

print("Word2Vec vocabulary length:", len(data2vec.wv.vocab))


data2vec.train(sentences)

data2vec.train(sentences, total_examples=data2vec.corpus_count, epochs=15)

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix = data2vec.wv.syn0

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[data2vec.wv.vocab[word].index])
            for word in data2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

    
points.head(10)
    
sns.set_context("poster")

points.plot.scatter("x", "y", s=10, figsize=(20, 12))
    
def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
    
    
plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))

plot_region(x_bounds=(1.0, 2.0), y_bounds=(10.0, 12.0))
plot_region(x_bounds=(20.0,30.0), y_bounds=(-10.0, -1.0))


data2vec.wv.most_similar('follow')

#lsi application
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

dct = Dictionary(sentences) 

corpus11 = [dct.doc2bow(line) for line in sentences] 
model = TfidfModel(corpus11)
corpus_tfidf = model[corpus11]


 doc_bow = [(0, 1), (0, 0)]
print(model[doc_bow])


for doc in corpus_tfidf:
    print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dct, num_topics=50)


corpus_lsi = lsi[corpus_tfidf]

lsi.print_topics(2)


