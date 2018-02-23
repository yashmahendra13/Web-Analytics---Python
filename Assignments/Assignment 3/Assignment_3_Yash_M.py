# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:46:04 2018

@author: ymahendr
"""
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd



def getReviews(movie_id):
    
    rows1=[]
    rows2=[]
    rows3=[]
    rows4=[]
    
    reviews=[]
    
    a = movie_id
    print("Movie Name", a)
    
    url = 'https://www.rottentomatoes.com/m/'+movie_id+'/reviews/?type=top_critics'
    source_code = requests.get(url)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text,'html.parser')
        
        
    for link in soup.find_all('a',{'class': 'unstyled bold articleLink'}):
        #href = link.get('href')
        title = link.string
        rows1.append(title)
        #print(title)
        
    for link in soup.find_all('div',{'class':'review_date subtle small'}):    
        date = link.string
        rows2.append(date)
        #print(date)
      
    for link in soup.find_all('div',{'class':'the_review'}):    
        review = link.string
        rows3.append(review)
        #print(review)
        
    for link in soup.find_all('div',{'class':'small subtle'}):
        text = link.get_text()
        a = text.strip().split("|")
        try:
            score = a[1]
        except IndexError:
            # not enough elements in the line
            continue
        
        rows4.append(score)
        #print(score)
#print(scores)
    
    for item1, item2, item3, item4 in zip(rows1, rows2, rows3, rows4):
        #print(item1, item2, item3, item4)
        reviews.append([item1,item2,item3,item4])
        
        #print(reviews)
    
    #print(reviews)
    return reviews

def mpg_plot():
    df = pd.read_csv('auto-mpg.csv')
    df.groupby(['model_year','origin'])["mpg"].mean().unstack()\
    .plot(kind='line', figsize=(8,4))
    plt.show()
        
getReviews('finding_dory')
if __name__ == "__main__":
    mpg_plot()
    movie_id='finding_dory'
    reviews=getReviews(movie_id)
    print(reviews) 