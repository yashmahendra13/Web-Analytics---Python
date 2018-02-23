# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Structure of your solution to Assignment 1 

import numpy as np
import csv

def count_token(text):
    
    token_count={}
    text = text.split()
    for index in range(len(text)):
        text[index] = text[index].strip()
        text[index] = text[index].lower()
        
    text = [x for x in text if len(x)>1] 
    
    
    token_count = ({x:text.count(x) for x in set(text)}) 
        
    return token_count

            # # The output of your text should be: 
            # {'this': 1, 'is': 1, 'example': 1, 'world!': 1, 'world': 1, 'hello': 2}

if __name__ == "__main__":
    text='''Hello world!
    This is a hello world example !'''  
                
    print(count_token(text))
    
     # Test Question 2
    #analyzer=Text_Analyzer("foo.txt", "foo.csv")
    vocabulary=analyzer.analyze()

class Text_Analyzer(object):
    
    def __init__(self, input_file, output_file):
        
        # add your code
        self.input_file = input_file
        self.output_file = output_file
          
    def analyze(self):
        
        
        # add your code
        #Reading the Input file
        
        input_file = self.input_file
        
        f = open(input_file, "r")                       
        # loop through all lines
        lines=[line for line in f]                     
        print (lines)

        #Concatenating into string
        sent_str = ""
        sent_str = ''.join(lines)
        print(sent_str)

        #calling the function "count_token" to get a token-count dictionary
        out_dict = count_token(sent_str)
        print(out_dict)

        #saving the dictionary into output_file with each key-value pair as a line delimited by comma
        with open('foo.csv','w') as f:
            w = csv.writer(f, delimiter=",")
            w.writerows(out_dict.items()) 
            
       
 


        
    