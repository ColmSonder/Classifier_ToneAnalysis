# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 08:15:21 2017

@author: HP-Colm
"""
## resources / guides
## https://raw.githubusercontent.com/minimaxir/stylistic-word-clouds/master/wordcloud_cnn.py
## http://minimaxir.com/2016/05/wordclouds/

import numpy as np
import random
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from palettable.colorbrewer.qualitative import Dark2_8
from palettable.colorbrewer.sequential import Blues_8, Reds_8
import os
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

stoplist = stopwords.words('english')
stoplist.append(unicode('subject'))
stoplist.append(unicode('nbsp'))



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
spam1 = init_lists(".../enron1/spam/")
ham1 = init_lists('..../enron1/ham/')
##spam2 and ham2 not used due to overlap with other data source used in MSc, the data itself is fine
spam3 = init_lists("..../enron3/spam/")
ham3 = init_lists('..../enron3/ham/')
ham4 = init_lists("..../enron4/ham/")
ham6 = init_lists('..../enron6/ham/')


spam1+=spam3
ham1+=ham3
ham1+=ham4
ham1 +=ham6

print(len(spam1))
print(len(ham1))
print(len(spam1)+len(ham1))


# PRE-PROCESS DATA
# breaks email into individual words, removes words on stoplist, lemmatizes words 
# to word stem removes "words" shorter than 2 letters i.e. !"£$% 

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence = word_tokenize(unicode(sentence, errors='ignore'))
    words = [i for i in sentence if i not in stoplist]
    lemon = [lemmatizer.lemmatize(word.lower()) for word in words if len(word) > 2]
    return [" ".join(lemon)]

# SPLIT PREPROCESSED EMAIL TEXT INTO SPAM AND HAM CLASSES TO VISUALISE EACH CLASS
spam = []
for i in spam1:
    j = preprocess(i)
    spam.append(j)
ham = []
for i in ham1:
    j = preprocess(i)
    ham.append(j)
print(len(ham))


def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return tuple(Blues_8.colors[random.randint(1,7)])


def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return tuple(Reds_8.colors[random.randint(1,7)])

# FILE PATH FOR THE ICON IMAGE TO BE USED IN VIZ, EXAMPLE HERE USED A STANDARD EMAIL ICON BUT ANY ICON 
# CAN BE USED FROM A SITE LIKE https://www.flaticon.com/ OR ANY OTHER    

fa_path = "...."
icon = "email-filled-closed-envelope"

spam_message = ''
for i in spam:
    for j in i :   
		spam_message += j

ham_message = ''
for i in ham:
    for j in i :   
		ham_message += j
		 		
# CHANGE FORMAT OF ICON TO JPG
# http://stackoverflow.com/questions/7911451/pil-convert-png-or-gif-with-transparency-to-jpg-without
icon_path = fa_path + "%s.png" % icon
mask = Image.new("RGBA", (516,516), (256,256,256,256))
icon = Image.open(icon_path).convert("RGBA")
x, y = icon.size
mask.paste(icon,(0,0,x,y),icon)
mask = np.array(mask)

# GENERATE WORD CLOUD
wc = WordCloud(background_color="white", max_words=2000, mask=mask,
               max_font_size=300, random_state=42)
# SPAM IMAGE       
wc.generate_from_text(spam_message)
wc.recolor(color_func=red_color_func, random_state=3)
wc.to_file("Enron_spam_wordcloud.png")

# HAM IMAGE
wc.generate_from_text(ham_message)
wc.recolor(color_func=blue_color_func , random_state=3)
wc.to_file("Enron_ham_wordcloud.png")