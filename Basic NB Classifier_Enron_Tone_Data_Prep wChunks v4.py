# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 15:07:52 2016

@author: HP-Colm
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:50:19 2016

@author: HP-Colm
"""

##https://blog.cambridgecoding.com/2016/01/25/implementing-your-own-spam-filter/
####https://pythonprogramming.net/sentiment-analysis-module-nltk-tutorial/
# https://bruceelgort.com/2016/06/07/using-ibm-watson-tone-analyzer-with-python/
# http://chardet.readthedocs.io/en/latest/


import time
import os
from watson_developer_cloud import ToneAnalyzerV3
import pandas as pd
import numpy as np
import sys


Load_start_time = time.time()
# GET DATASET
#  iteratively read the files in folders
def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r')
        a_list.append(f.read())
    f.close()
    return a_list

#  create spam and ham lists
#spam1 = init_lists("C:/Users/HP-Colm/Documents/Data Analytics/Thesis/BasicNBClassifier/enron1/spam/")
#ham1 = init_lists('C:/Users/HP-Colm/Documents/Data Analytics/Thesis/BasicNBClassifier/enron1/ham/')
#spam3 = init_lists("C:/Users/HP-Colm/Documents/Data Analytics/Thesis/BasicNBClassifier/enron3/spam/")
#ham3 = init_lists('C:/Users/HP-Colm/Documents/Data Analytics/Thesis/BasicNBClassifier/enron3/ham/')
#ham4 = init_lists("C:/Users/HP-Colm/Documents/Data Analytics/Thesis/BasicNBClassifier/enron4/ham/")
ham6 = init_lists('C:/Users/HP-Colm/Documents/Data Analytics/Thesis/BasicNBClassifier/enron6/ham/')

# combine two lists keeping the labels. 
# create a list of two-place tuples – Python objects, where the first member of the 
# pair stores the text of the email and the second one – its label
#all_emails = [(unicode(email, encoding = 'ISO-8859-2'), 'spam') for email in spam1]
#all_emails = [(unicode(email, encoding = 'ISO-8859-2'), 'ham') for email in ham1]
#all_emails = [(unicode(email, encoding = 'ISO-8859-2'), 'spam') for email in spam3]
#all_emails = [(unicode(email, encoding = 'ISO-8859-2'), 'ham') for email in ham3]
#all_emails = [(unicode(email, encoding = 'ISO-8859-2'), 'ham') for email in ham4]
all_emails = [(unicode(email, encoding = 'ISO-8859-2'), 'ham') for email in ham6]               

#sum_ham = float(len(ham1)) 
#sum_ham = float(len(ham3)) 
#sum_ham = float(len(ham4)) 
sum_ham = float(len(ham6)) 

"""+ float(len(ham4)) + float(len(ham6))"""
print(sum_ham)

#sum_spam = float(len(spam1)) 
#sum_spam = float(len(spam3))
#print(sum_spam)              
#print(sum_ham + sum_spam)               
print (len(all_emails))

#del spam1[:]
#del ham1[:]
#del spam3[:]
#del ham3[:]
#del ham4[:]
del ham6[:]

# blob = open('C:/Users/HP-Colm/Documents/Python Scripts/positive.txt').read() 
# print(chardet.detect(blob))
print "Loading took", time.time() - Load_start_time, "to run"

enron1_df = pd.DataFrame(all_emails,columns=["email", "label"])
numchunks = int(np.ceil(float(len(all_emails))/500))

# This function creates chunks and returns them
def chunkify(lst,n):
    return [ lst[i::n] for i in xrange(n) ]

chunks = chunkify(all_emails, numchunks)
print(len(chunks))

# Get Tone Data
tone_job_start_time = time.time()
tone_analyzer = ToneAnalyzerV3(
   username='562e682b-0cbd-4bb0-ac72-ed77bc3729a8',
   password='JRFzwfvFxl66',
   version='2016-05-19 ')    

ibm_results = []

for chunk in chunks:
    for e_mail, label in chunk:
        if len(e_mail) == 0:
            results = tone_analyzer.tone(text = ('0'), sentences = False)
            if results != False:
                ibm_results.append(results)
                print(len(ibm_results))
            else:
                print("Something went wrong")
        elif sys.getsizeof(e_mail)> 120000:
            results = tone_analyzer.tone(text = ('0'), sentences = False)
            if results != False:
                ibm_results.append(results)
                print(len(ibm_results))
            else:
                print("Something went wrong")
        elif e_mail.count(".")+e_mail.count("!")+e_mail.count("?")+e_mail.count("\n") >900:
            results = tone_analyzer.tone(text = ('0'), sentences = False)
            if results != False:
                ibm_results.append(results)
                print(len(ibm_results))
            else:
                print("Something went wrong")
        else:
            results = tone_analyzer.tone(text = (e_mail), sentences = False)
            if results != False:
                ibm_results.append(results)
                print(len(ibm_results))
            else:
                print("Something went wrong")
                     

print "Tone Analyser Processing took", time.time() - tone_job_start_time, "to run"
tone_features_list = time.time()
tone_features = []
for results in ibm_results:
    tone_feature = (
          results['document_tone']['tone_categories'][0]['tones'][0]['score'],
          results['document_tone']['tone_categories'][0]['tones'][1]['score'],
          results['document_tone']['tone_categories'][0]['tones'][2]['score'],
          results['document_tone']['tone_categories'][0]['tones'][3]['score'],
          results['document_tone']['tone_categories'][0]['tones'][4]['score'],
          results['document_tone']['tone_categories'][1]['tones'][0]['score'],
          results['document_tone']['tone_categories'][1]['tones'][1]['score'],
          results['document_tone']['tone_categories'][1]['tones'][2]['score'],
          results['document_tone']['tone_categories'][2]['tones'][0]['score'],
          results['document_tone']['tone_categories'][2]['tones'][1]['score'],
          results['document_tone']['tone_categories'][2]['tones'][2]['score'],
          results['document_tone']['tone_categories'][2]['tones'][3]['score'],
          results['document_tone']['tone_categories'][2]['tones'][4]['score'])
    tone_features.append(tone_feature)     
print "Tone Feature list creation took", time.time() - tone_features_list, "to run" 


tone_features_dataframe = time.time()

enron1t_df = pd.DataFrame(tone_features, index = None ,columns=["t_Anger","t_Disgust", "t_Fear","t_Joy","t_Sadness","t_Analytical","t_Confident","t_Tentative","t_Openness","t_Conscientiousness","t_Extraversion","t_Agreeableness","t_EmotionalRange"]) 
frames = [enron1_df , enron1t_df]
enron1tone_df = pd.concat(frames, axis = 1)
enron1tone_df.fillna(value = 0)


print "Tone Feature dataframe creation took", time.time() - tone_features_dataframe, "to run" 
print "No of emails in all_emails list", len(all_emails)
print "No of rows in all_emails dataframe", len(enron1_df)
print "No of tone feature sets created", len(tone_features)
print "No of rows in enron tone features dataframe", len(enron1t_df)

enron1tone_df.to_excel('C:/Users/HP-Colm/Documents/Data Analytics/Thesis/BasicNBClassifier/enron1/enron1_dataset_w_tonefeats5.xlsx', sheet_name='Sheet1')
