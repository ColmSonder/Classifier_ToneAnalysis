# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 15:07:52 2016

@author: HP-Colm
"""


# resources/guidelines
# https://bruceelgort.com/2016/06/07/using-ibm-watson-tone-analyzer-with-python/


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

#  create spam and ham lists from enron data set
spam1 = init_lists("..../enron1/spam/")
ham1 = init_lists('..../enron1/ham/')

# combine two lists keeping the labels. 
# create a list of two-place tuples – Python objects, where the first member of the 
# pair stores the text of the email and the second one – its label
spam_labelled = [(unicode(email, encoding = 'ISO-8859-2'), 'spam') for email in spam1]
ham_labelled = [(unicode(email, encoding = 'ISO-8859-2'), 'ham') for email in ham1]
all_emails = spam_labelled.extend(ham_labelled)

#check
print (len(spam_labelled))
print (len(ham_labelled))            
print (len(all_emails))

print "Loading took", time.time() - Load_start_time, "to run"

# the IBM Watson Tone Classifier is throttled so it will only take a certain volume at any time 
# so the list needs to be broken into chunks that are sent to it otherwise it will return an error.

#  CREATE A DATAFRAME OF THE EMAIL TEST AND CHUNK UP DATAFRAME INTO SMALLER BATCHES
enron1_df = pd.DataFrame(all_emails,columns=["email", "label"])
numchunks = int(np.ceil(float(len(all_emails))/500))

# This function creates chunks and returns them
def chunkify(lst,n):
    return [ lst[i::n] for i in xrange(n) ]

chunks = chunkify(all_emails, numchunks)
print(len(chunks))

# GET TONE DATA ON EMAIL TEXT FROM IBM WATSON
tone_job_start_time = time.time()
tone_analyzer = ToneAnalyzerV3(
   username='INSERT CREDENTIALS',
   password='INSERT CREDENTIALS',
   version='INSERT CURRENT VERSION ')    

# CREATE A LIST TO HOLD THE RETURNED VALUES FROM IBM WATSON TONE CLASSIFIER
ibm_results = []

# CHECK EACH CHUNK FOR DATA THAT CAUSES FAILURES FROM IBM WATSON TONE CLASSIFIER NAMELY
# NO TEXT IN LIST ITEM
# AMOUNT OF TEXT IS TOO LARGE FOR TONE CLASSIFIER TO RETURN RESULTS
# TOO MANY SYMBOLS
# this routine outputs a count just to show progress as the service can be slow and can time-out so its 
# a guage of how far the process has gotten

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

# EXTRACT TONE FEATURES FROM RETURNED JSON RESULTS
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

# ADD RESULTS TO DATAFRAME OF TEXT AND LABELS
enron1t_df = pd.DataFrame(tone_features, index = None ,columns=["t_Anger","t_Disgust", "t_Fear","t_Joy","t_Sadness","t_Analytical","t_Confident","t_Tentative","t_Openness","t_Conscientiousness","t_Extraversion","t_Agreeableness","t_EmotionalRange"]) 
frames = [enron1_df , enron1t_df]
enron1tone_df = pd.concat(frames, axis = 1)
enron1tone_df.fillna(value = 0)


print "Tone Feature dataframe creation took", time.time() - tone_features_dataframe, "to run"

#CHECK TOTALS 
print "No of emails in all_emails list", len(all_emails)
print "No of rows in all_emails dataframe", len(enron1_df)
print "No of tone feature sets created", len(tone_features)
print "No of rows in enron tone features dataframe", len(enron1t_df)

# EXPORT TO EXCEL
enron1tone_df.to_excel('C:/Users/HP-Colm/Documents/Data Analytics/Thesis/BasicNBClassifier/enron1/enron1_dataset_w_tonefeats5.xlsx', sheet_name='Sheet1')
