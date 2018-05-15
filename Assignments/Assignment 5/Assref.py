# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:43:25 2018

@author: ymahendr
"""

import nltk, string
#from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
import numpy as np  
import pandas as pd
import csv
from nltk.stem.porter import PorterStemmer
from scipy.spatial import distance

# Step 1. get tokens of each document as list
def get_doc_tokens(doc):
    stop_words = stopwords.words('english')
    
    tokens=[token.strip() \
            for token in nltk.word_tokenize(doc.lower()) \
            if token.strip() not in stop_words and\
               token.strip() not in string.punctuation]
    
    # you can add bigrams, collocations, stemming, 
    # or lemmatization here
    #if normalize == 'Stem':
    #    porter_stemmer = PorterStemmer()
    #    tokens = porter_stemmer.stem(tokens)
    #print(tokens)
        
    token_count={token:tokens.count(token) for token in set(tokens)}
    return token_count

def tfidf(docs,normal):
    # step 2. process all documents to get list of token list
    porter_stemmer = PorterStemmer()
    
    docs_tokens={idx:get_doc_tokens(doc) \
             for idx,doc in enumerate(docs)}

    if normal == 'Stem':
        porter_stemmer = PorterStemmer()
        docs_tokens = porter_stemmer.stem(docs_tokens)
        print(docs_tokens)
    
    # step 3. get document-term matrix
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
    dtm=dtm.fillna(0)
      
    # step 4. get normalized term frequency (tf) matrix        
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf.T, doc_len).T
    
    # step 5. get idf
    df=np.where(tf>0,1,0)
    #idf=np.log(np.divide(len(docs), \
    #    np.sum(df, axis=0)))+1

    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1    
    smoothed_tf_idf=tf*smoothed_idf
    
    return smoothed_tf_idf



if __name__ == "__main__":  
    
    # test collocation
    #text=nltk.corpus.reuters.raw('test/14826')
    #tokens=nltk.word_tokenize(text.lower())
    #print(top_collocation(tokens, 10))
    
    
    # load data
    docs=[]
    with open("amazon_review_300.csv","r") as f:
        reader=csv.reader(f)
        for line in reader:
            docs.append(line[2])
    
    smoothed_tf_idf = tfidf(docs,'None')
    #print(smoothed_tf_idf)
    #print(len(smoothed_tf_idf))
    
    # Find similar documents -- No STEMMING
    similarity=1-distance.squareform\
    (distance.pdist(smoothed_tf_idf, 'cosine'))
    similarity

    # find top doc similar to first one
    print(np.argsort(similarity)[:,::-1][0,0:6])
    a = np.argsort(similarity)[:,::-1][0,0:6]
    
    for x in a:
        print(x,'',docs[x])
    
    #for idx, docs in enumerate(docs):
    #    for idx in a:
    #        print(idx,docs)
    # Find similar documents -- STEMMING 
    