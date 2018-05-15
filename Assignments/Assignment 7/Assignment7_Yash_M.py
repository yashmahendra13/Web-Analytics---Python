# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:48:25 2018

@author: ymahendr
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_json('ydata_3group.json')



text=list(df[0])
target=list(df[1])

# initialize the TfidfVectorizer 
# set min document frequency to 5

tfidf_vect = TfidfVectorizer(stop_words="english",\
                             min_df=60,max_df=0.90) 

# generate tfidf matrix
dtm= tfidf_vect.fit_transform(text)
#print (dtm.shape)



# set number of clusters
num_clusters=3

# initialize clustering model
# using cosine distance
# clustering will repeat 20 times
# each with different initial centroids
clusterer = KMeansClusterer(num_clusters, \
                            cosine_distance, repeats=20)

# samples are assigned to cluster labels starting from 0
clusters = clusterer.cluster(dtm.toarray(), \
                             assign_clusters=True)

#print the cluster labels of the first 5 samples
#print(clusters[0:5])

df1=pd.DataFrame(list(zip(target, clusters)), \
                columns=['actual_class','cluster'])
df1.head(40)
pd.crosstab(index=df1.cluster, columns=df1.actual_class)

cluster_dict={0:'T3', 2:"T2",\
              1:'T1'}

# Assign true class to cluster
predicted_target=[cluster_dict[i] for i in clusters]

print(metrics.classification_report\
      (target, predicted_target))


centroids=np.array(clusterer.means())

sorted_centroids = centroids.argsort()[:, ::-1] 

voc_lookup= tfidf_vect.get_feature_names()

for i in range(num_clusters):
    
    # get words with top 20 tf-idf weight in the centroid
    top_words=[voc_lookup[word_index] \
               for word_index in sorted_centroids[i, :20]]
    print("Cluster %d: %s " % (i, "; ".join(top_words)))
    
    
#Cluster 0 (T3) : Travel
#Cluster 1 (T1) : Bad News Natural and Manmade
#Cluster 2 (T2) : Finance

#Task 2 LDA

# Exercise 5.2. Preprocessing - Create Term Frequency Matrix


# LDA can only use raw term counts for LDA 
tf_vectorizer = CountVectorizer(min_df=60,max_df=0.9, stop_words='english')
tf = tf_vectorizer.fit_transform(text)

# each feature is a word (bag of words)
# get_feature_names() gives all words
tf_feature_names = tf_vectorizer.get_feature_names()

#print(tf_feature_names[0:10])
#print(tf.shape)

num_topics = 3

lda = LatentDirichletAllocation(n_components=num_topics, \
                                max_iter=20,verbose=1,
                                evaluate_every=1, n_jobs=1,
                                random_state=0).fit(tf)

# Exercise 5.5. Assign documents to topic


# Generate topic assignment of each document
topic_assign=lda.transform(tf)
#print(topic_assign[0:5])

topics=np.copy(topic_assign)

i=0
while i < 4021:
    topics[i]=np.where(topics[i]==topics[i].max(), 1, 0)
    i=i+1
#print(topics[0:5])
topics[0:5]

i = 0
while i < 4021:
    if topics[i,1]==1:
        topics[i,1]='2'
    
    i=i+1

i = 0
while i < 4021:
    if topics[i,2]==1:
        topics[i,2]=3
    
    i=i+1

res = [sum(e) for e in topics]
#print(res)
print(len(res))

i = 0
while i < 4021:
    if res[i]==1:
        res[i]='T1'
    elif res[i]==2:
        res[i]='T2'
    elif res[i]==3:
        res[i]='T3'
    
    i=i+1
    
from sklearn import metrics
print(metrics.classification_report\
      (target, res))
















    



