#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import re
from gensim import corpora, models
import gensim
from nltk import bigrams 
from nltk import trigrams
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob as tb
from datetime import datetime, timedelta
import time
import os


# In[2]:


path_to_json='Lexis_Data'
header=True


# In[3]:


import subprocess
 
def run_cmd(args_list):
        print('Running system command: {0}'.format(' '.join(args_list)))
        proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        s_output, s_err = proc.communicate()
        s_return =  proc.returncode
        return s_return, s_output, s_err 


# In[21]:


get_ipython().run_cell_magic('bash', '', '\ncd Lexis_Data\nls > names.txt')


# In[27]:


file_names = []

with open('Lexis_Data/names.txt', 'r') as f:
    x = f.readlines()
    
file_names = []

for i in x:
    file_names.append(i[:-1])
    
file_names = file_names[:-1]


# In[28]:


file_names


# In[4]:


# Files=['lexis_sb_20190123T0000','lexis_sb_20190124T0000','lexis_sb_20190125T0000'] 
Files = file_names
########################Reading the data into a list################################
lexis_news=pd.DataFrame()
for file in Files:
    (ret, out, err)= run_cmd(['hadoop', 'fs', '-get', '/data/atl_sprint_2018/lexis_archive_sb/%s.json'
                          %file, 'Lexis_Data/'])
    lexis_data=[]
    with open("%s/%s.json" %(path_to_json,file),'r') as f:
        for line in f:
            try:
                temp=json.loads(line)
                lexis_data.append(temp)
            except:
                continue       
#####################Creating the base dataframe for lexis data##############################
    lexis_temp=pd.DataFrame()
    lexis_temp['URL']=list(map(lambda x:x['URL'],lexis_data))
#     lexis_temp['City']=list(map(lambda x:x['City'],lexis_data))
#     lexis_temp['Country']=list(map(lambda x:x['Country'],lexis_data))
#     lexis_temp['State']=list(map(lambda x:x['State'],lexis_data))
#     lexis_temp['Location']=list(map(lambda x:x['Location'],lexis_data))
#     lexis_temp['Description']=list(map(lambda x:x['Description'],lexis_data))
       
    lexis_temp['Original_Date']=list(map(lambda x:x['Date'],lexis_data))
#     lexis_temp['Language']=list(map(lambda x:x['Language'],lexis_data))
    lexis_temp['Source']=list(map(lambda x:x['Original Source'],lexis_data))
    lexis_temp['Headline']=list(map(lambda x:x['Headlines'],lexis_data))
    #lexis_temp['Text']=list(map(lambda x:x['Text'],lexis_data))
    lexis_temp['Text']=list(map(lambda x:" ".join(x['Text'].split()),lexis_data))
    
    lexis_temp['Length_Category']=list(map(lambda x: "<50" if len(x['Text'].split())< 50 else "50-100"
                                             if len(x['Text'].split())>= 50 and len(x['Text'].split())<100 else "100-200"
                                             if len(x['Text'].split())>= 100 and len(x['Text'].split())<200 else ">=200",lexis_data))
        
#######################Converting None dates to the actual date##############################
    temp=[]
    d=lexis_temp.groupby('Original_Date').Text.count().reset_index(name='Volume')
    d=d[d.Original_Date != 'None']['Original_Date'][0]
    temp=[d]*len(lexis_temp)
    lexis_temp['Date']=temp
    lexis_news=lexis_news.append(lexis_temp)   



# In[5]:


############################Removing Duplicates################################################
        
lexis_news=lexis_news.drop_duplicates(subset=['Text'],keep='first',inplace=False).reset_index(drop=True)
        


# In[6]:


########################Getting the info in a .CSV file###############################
        
lexis_news[lexis_news.Text !='None'].to_csv("News_Labelling_Data.csv",sep=",",header=True,mode='w',index=True)


# In[ ]:





# In[7]:


from gensim.summarization.summarizer import summarize


# In[9]:


lexis_news_list2=[]
working_counter = 0
for text in lexis_news['Text']: 
    text = str(text).replace('\'', '\\\'')
    text = str(text).replace('?', '? ')
    text = str(text).replace(';', '.')
    #print(text)
    if len(str(text).split('.'))>1 or len(str(text).split('. '))>1 or len(str(text).split('.\n'))>1 :

        try:
            lexis_news_list2.append(summarize(text))
            working_counter+=1
        except:
            
            try:
                lexis_news_list2.append(summarize(text, ratio = 0.5))
                
            except:
                lexis_news_list2.append(' ')
    else:
        lexis_news_list2.append(' ')

lexis_news['Clean Text']=lexis_news_list2
# print(working_counter)
# print(counter)
lexis_news.to_csv("Lexis_Summarized_Data.csv", index=False)


# In[ ]:




