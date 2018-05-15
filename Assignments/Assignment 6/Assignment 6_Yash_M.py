# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:34:22 2018

@author: ymahendr
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import precision_recall_fscore_support
#from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import matplotlib.pyplot as plt



df1 = pd.read_csv('amazon_review_300.csv', delimiter=',',names = ["Labels", "Title", "Reviews"])

# convert df to list
text=list(df1['Reviews'])
#print(text[0])
target=list(df1['Labels'])
#print(target[0])

#Answer 1
# initialize the TfidfVectorizer 
print('Answer 1\n')

tfidf_vect = TfidfVectorizer() 

# with stop words removed
tfidf_vect = TfidfVectorizer(stop_words="english") 

# generate tfidf matrix
dtm= tfidf_vect.fit_transform(text)

print("type of dtm:", type(dtm))
print("size of tfidf matrix:", dtm.shape)

# Classification using a six fold

metrics = ['precision_macro', 'recall_macro', "f1_macro"]

#clf = MultinomialNB()
clf = MultinomialNB(alpha=0.5)

cv = cross_validate(clf, dtm, target, scoring=metrics, cv=6, return_train_score=True)
print("Test data set average precision:")
print(cv['test_precision_macro'])
print("\nTest data set average recall:")
print(cv['test_recall_macro'])
print("\nTest data set average fscore:")
print(cv['test_f1_macro'])

# To see the performance of training data set use 
# cv['train_xx_macro']
print("\nTrain data set average fscore:")
print(cv['train_f1_macro'])
print('\n')

#Answer 2
print('Answer 2\n')

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', MultinomialNB())
                   ])


parameters = {'tfidf__min_df':[1,2,3,5],
              'tfidf__stop_words':[None,"english"],
              'clf__alpha': [0.5,1.0,1.5,2.0],
}

# the metric used to select the best parameters
metric =  "f1_macro"

# GridSearch also uses cross validation
gs_clf = GridSearchCV\
(text_clf, param_grid=parameters, scoring=metric, cv=6)

# due to data volume and large parameter combinations
# it may take long time to search for optimal parameter combination
# you can use a subset of data to test
gs_clf = gs_clf.fit(text, target)


for param_name in gs_clf.best_params_:
    print(param_name,": ",gs_clf.best_params_[param_name])

print("best f1 score:", gs_clf.best_score_)

#Answer 3
print('Answer 3\n')

df2 = pd.read_csv('amazon_review_large.csv', delimiter=',',names = ["Labels", "Reviews"])

# convert df to list
text=list(df2['Reviews'])
#print(text[0])
target=list(df2['Labels'])
#print(target[0])

def NaiveBayesOP(text,target):
    text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=None)),
                     ('clf1', MultinomialNB()),       
                        ])
    
    cv = cross_validate(text_clf, text, target, scoring=metrics, cv=10)
    
    #print("\nTest data set average fscore:")
    #print(cv['test_f1_macro'])
    f1 = cv['test_f1_macro']
    return f1

def SVCOP(text,target):
    text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=None)),
                     ('clf1', svm.LinearSVC()),       
                        ])
    
    cv = cross_validate(text_clf, text, target, scoring=metrics, cv=10)
    
    #print("\nTest data set average fscore:")
    #print(cv['test_f1_macro'])
    f1 = cv['test_f1_macro']
    return f1

metrics = ['precision_macro', 'recall_macro', "f1_macro"]
Classifier1 = []
Classifier2 = []
i = 400
a = []
b = []
    
    
while i<=20000:     
    a = SVCOP(text[0:i],target[0:i])
    a = np.mean(a)
    Classifier1.append(a)
        
    b = NaiveBayesOP(text[0:i],target[0:i])
    b = np.mean(b)
    Classifier2.append(b)
    i=i+400
        
print('\nResults of Classifier 1')
print(Classifier1)
print('\nResults of Classifier 2')
print(Classifier2)

#Line Chart
#Plot 1
x = range(400,20400,400)
x = list(x)
plt.figure(figsize=(12,8))
plt.xlabel('Sample Size')
plt.ylabel('F-Score')
plt.grid(True)
plt.title('Support Vector Machine')
plt.plot(x,Classifier1)

#Plot2
plt.figure(figsize=(12,8))
plt.xlabel('Sample Size')
plt.ylabel('F-Score')
plt.grid(True)
plt.title('Naive Bayes')
plt.plot(x,Classifier2)

#Plot3
plt.figure(figsize=(12,8))
plt.xlabel('Sample Size')
plt.ylabel('F-Score')
plt.grid(True)
plt.title('Support Vector Machine vs Naive Bayes')
plt.plot(x,Classifier1)
plt.plot(x,Classifier2)
plt.gca().legend(('SVM','NB'))












