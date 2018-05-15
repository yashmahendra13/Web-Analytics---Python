# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:19:33 2018

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

#Answer 1
def top_collocation(tokens, K):
    result=[]
    k = K
    # add your code here
    tagged_tokens= nltk.pos_tag(tokens)
    #print(len(tagged_tokens))
    
    bigrams=list(nltk.bigrams(tagged_tokens))
    #print(bigrams)
    
    freq = nltk.FreqDist(bigrams)
    print(freq)
    
    phrases1 =[ (x[0],y[0]) for (x,y) in bigrams \
         if x[1].startswith('JJ')  \
         and y[1].startswith('NN')]
    #print(len(phrases1))
    #print(phrases1)
    
    phrases2 =[ (x[0],y[0]) for (x,y) in bigrams \
         if x[1].startswith('NN') \
         and y[1].startswith('NN')]
    #print(len(phrases2))
    #print(phrases2)
    
    phrases = phrases1 + phrases2
    
    freq10 = nltk.FreqDist(phrases)
    freq10 = sorted(freq10.items(), key=lambda x: x[1], reverse=True)
    result = freq10[0:k]
    
    return result

#Answer 2
# Step 1. get tokens of each document as list
def get_doc_tokens(doc,normal):
    
    stop_words = stopwords.words('english')
    porter_stemmer = PorterStemmer()
    
    if normal == 'Stem':
            tokens=[porter_stemmer.stem(token.strip()) \
            for token in nltk.word_tokenize(doc.lower()) \
            if token.strip() not in stop_words and\
               token.strip() not in string.punctuation]
    else:
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
    docs_tokens={idx:get_doc_tokens(doc,normal) \
             for idx,doc in enumerate(docs)}

    #if normalize == 'Stem':
    #    porter_stemmer = PorterStemmer()
    #    docs_tokens = porter_stemmer.stem(docs_tokens.values)
    #print(docs_tokens)
    
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
    text=nltk.corpus.reuters.raw('test/14826')
    tokens=nltk.word_tokenize(text.lower())
    print(top_collocation(tokens, 10))
    
    # load data
    docs=[]
    with open("amazon_review_300.csv","r") as f:
        reader=csv.reader(f)
        for line in reader:
            docs.append(line[2])
    
    smoothed_tf_idf1 = tfidf(docs,'None')
    print('')
    print("No-Stem TF_IDF")
    print(smoothed_tf_idf1)
    print(len(smoothed_tf_idf1))
    
    smoothed_tf_idf2 = tfidf(docs,'Stem')
    print('')
    print("Stem TF_IDF")
    print(smoothed_tf_idf2)
    print(len(smoothed_tf_idf2))
    
    # Find similar documents -- No STEMMING
    similarity1=1-distance.squareform\
    (distance.pdist(smoothed_tf_idf1, 'cosine'))
    similarity1

    # find top doc similar to first one
    print('')
    print('Simmilar documents -- No STEMMING')
    print(np.argsort(similarity1)[:,::-1][0,0:6])
    a = np.argsort(similarity1)[:,::-1][0,0:6]
    
    for x in a:
        print(x,'',docs[x])
    
    #for idx, docs in enumerate(docs):
    #    for idx in a:
    #        print(idx,docs)
    
    # Find similar documents -- STEMMING 
    similarity2=1-distance.squareform\
    (distance.pdist(smoothed_tf_idf2, 'cosine'))
    similarity2

    # find top doc similar to first one
    print('')
    print('Simmilar documents -- STEMMING')
    print(np.argsort(similarity2)[:,::-1][0,0:6])
    a = np.argsort(similarity2)[:,::-1][0,0:6]
    
    for x in a:
        print(x,'',docs[x])    
    
    
    