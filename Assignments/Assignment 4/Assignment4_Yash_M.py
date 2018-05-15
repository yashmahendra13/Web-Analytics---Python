# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:54:08 2018

@author: ymahendr
"""
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords



def tokenize(text):
    
    tokens=[]
    
    # write your code here
    #input = text.lower()
    input=nltk.word_tokenize(text.lower())
    #token = re.search(r'^[a-z][a-z\-\_]*[^\-\_]$',input)
    token=[t for t in input if re.search(r'^[a-z][a-z\-\_]*[^\-\_]$',t)]
    stop_words = stopwords.words('english')
    tokens = [word for word in token if word not in stop_words]
    return tokens

def sentiment_analysis(text, positive_words, negative_words):
    
    sentiment=None
    
    # write your code here
        # write your code here
    #Tokenize
    #print(text)
    negations=['not', 'too', 'n\'t', 'no', 'cannot', 'neither','nor']
    tokens = nltk.word_tokenize(text)    
    #print(len(tokens) )                   
    #print (tokens)     
    positive_tokens=[]
    negative_tokens=[]
                      

    for idx, token in enumerate(tokens):
        if token in positive_words:
            if idx>0:
                if tokens[idx-1] not in negations:
                    positive_tokens.append(token)
            else:
                positive_tokens.append(token)

    for idx, token in enumerate(tokens):
        if token in negative_words:
            if idx>0:
                if tokens[idx-1] in negations:
                    positive_tokens.append(token)

            
    #print(positive_tokens)
    #print(len(positive_tokens))

    for idx, token in enumerate(tokens):
        if token in negative_words:
            if idx>0:
                if tokens[idx-1] not in negations:
                    negative_tokens.append(token)
                else:
                    negative_tokens.append(token)
                
    for idx, token in enumerate(tokens):
        if token in positive_words:
            if idx>0:
                if tokens[idx-1] in negations:
                    negative_tokens.append(token)
        
    #positive_tokens=[token for token in tokens \
    #            if token in positive_words]
    #print(positive_tokens)
    #len(positive_tokens)
    
    #negative_tokens=[token for token in tokens \
    #             if token in negative_words]

    #print(negative_tokens)
    #len(negative_tokens)
    
    if len(positive_tokens)>len(negative_tokens):
        #print("Positive")
        sentiment = "Positive"
    elif len(positive_tokens)<=len(negative_tokens):
        #print("Negative")
        sentiment = "Negative"
        
    return sentiment


def performance_evaluate(input_file, positive_words, negative_words):
    
    accuracy=None

    # write your code here
    
    df = pd.read_csv(input_file, delimiter=',',names = ["Neg or Pos", "Title", "Reviews"])
    df.drop('Title', axis=1, inplace=True)
    # Or export it in many ways, e.g. a list of tuples
    tuples = [tuple(x) for x in df.values]
    #text = tuples[2][1]
    #answer = sentiment_analysis(text,positive_words,negative_words)
    #sentiment_analysis()
    answer = []
    #temp = sentiment_analysis(text,positive_words,negative_words)
    #answer.append(temp)
    for i in range(0,300):
        text = tuples[i][1]
        temp = sentiment_analysis(text,positive_words,negative_words)
        if(temp=='Positive'):
            answer.append(2)
        else:
            answer.append(1)
    

    #print(answer)
    #len(answer)
    count = 0
    for i in range(0,300):
        if answer[i] == tuples[i][0]:
            count = count+1

    #print(count)
    accuracy = count/300 
    return accuracy

if __name__ == "__main__":  
    
    text="this is a breath-taking ambitious movie; test text: abc_dcd abc_ dvr89w, abc-dcd -abc"

    tokens=tokenize(text)
    print("tokens:")
    print(tokens)
    
    
    with open("positive-words.txt",'r') as f:
        positive_words=[line.strip() for line in f]
        
    with open("negative-words.txt",'r') as f:
        negative_words=[line.strip() for line in f]
        
    print("\nsentiment")
    sentiment=sentiment_analysis(text, positive_words, negative_words)
    print(sentiment)
    
    accuracy=performance_evaluate("amazon_review_300.csv", positive_words, negative_words)
    print("\naccuracy")
    print(accuracy)
