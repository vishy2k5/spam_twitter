# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:56:09 2019

@author: Vishy
"""
q=[]
for i in lsa_clean_list[0]:
    print(i)
    q.append(data2vec.wv.most_similar(i))
    
    
q=data2vec.most_similar('free')


with open('smili.txt', 'w') as f:
    for item in q:
        f.write(str(item))
    

with open('smili.txt', 'r') as myfile:
    datavec=myfile.read().replace('\n', '')





raw_vec = tokenizer.tokenize(datavec)
    
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

vecclean = []
for i in raw_vec:
    if len(i) > 0:
        vecclean.append(str(sentence_to_wordlist(i)))






with open('similar.txt', 'w') as f:
    for item in lsa_clean_list:
        f.write(str(item))









