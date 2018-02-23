# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:29:34 2018

@author: ymahendr
"""

import numpy as np
import pandas as pd

def analyze_tf(arr):
    
    tf_idf=None
    print('Input array')
    print(arr)
    print('\n')
    #normalizing the frequency of each word
    b = np.sum(arr,axis=1)
    Z = arr.T / b
    tf = Z.T
    print('Term Frequency - tf')
    print(tf)
    print('\n')
    
    #calculating the document frequency (df) of each word
    bf = np.where(tf>0,1,0)
    df = np.sum(bf, axis=0)
    print('Document Frequency - df')
    print(df)
    print('\n')
    
    #calculating tf_idf array
    tf_idf = tf/ df
    print('Inverse Document frequency - tf_idf')
    print(tf_idf)
    print('\n')
    
    #printing the top 3 largest values in the tf_idf array
    x1 = np.argsort(tf_idf)
    print('Printing top 3 largest values in tf_idf')
    print(x1[:,-3:])
    print('\n')
    return tf_idf


def analyze_cars():
    
    #Reading the csv file
    df = pd.read_csv("cars.csv")
    print('Head and tail of CSV File')
    print('\n')
    print(df.head(2))
    print('\n')
    print(df.tail(2))
    print('\n')
    
    #Sort the data by "cylinders" and "mpg" in decending order and report the first three rows after sorting
    #print("\nsort the values by columns Cylinders and mpg")
    df1 = df.sort_values(by=['cylinders','mpg'], ascending=False)
    print('Reporting the first three rows after sorting')
    print('\n')
    print(df1.head(3))
    print('\n')
    
    #Create a new column called "brand" to store the brand name as the first word in "car" column
    df['brand'] = df['car'].apply(lambda col: col.split()[0])
    print('Dataframe with new brand coloumn')
    print('\n')
    print(df.head(2))
    print('\n')
    
    #Show the mean, min, and max acceleration values by "cylinders" for each of these brands: "ford", "buick" and "honda"
    y = df[df.brand.isin(['ford','buick','honda'])]
    print('Mean, min, and max acceleration values by "cylinders" for each of these brands: "ford", "buick" and "honda')
    print('\n')
    print(pd.pivot_table(data=y, values=['acceleration'], index=['cylinders'], columns=['brand'], aggfunc=(np.mean,np.min,np.max)))
    print('\n')
    
    #Create a cross tab to show the average mpg of each brand and each clinder value. Use "brand" as row index 
    #and "clinders" as column index.
    print('Cross tab')
    print('\n')
    print(pd.crosstab(index=df.brand, columns=df.cylinders, values=df.mpg, aggfunc=np.mean))
    print('\n')
       
    # add your code
    
    
if __name__ == "__main__":  
    
    #1 Test Question 1
    arr=np.random.randint(0,3,(4,8))

    tf_idf=analyze_tf(arr)
    
    # Test Question 2
    analyze_cars()