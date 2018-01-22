# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:50:19 2016

@author: HP-Colm
"""
# This code takes the public dataset of Enron emails that was prepared for the paper:
# Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006, July). Spam filtering with 
# naive bayes-which naive bayes?. CEAS, 17, 28-69. and 1) preps the data, 2) then applies ML 
# techniques to classify the emails as spam or ham 3) evaluates the classifers success
# using average class accuracy outputs and lastly 4) the results as confusion 
# matrices.
# Dataset was available in May 2017 from www.csmining.org as 6 folders of spam and ham emails

# related blog posts / tutorials that guided the below
##https://blog.cambridgecoding.com/2016/01/25/implementing-your-own-spam-filter/
##https://pythonprogramming.net/sentiment-analysis-module-nltk-tutorial/

# import packages
import os
import random
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import operator
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
stoplist = stopwords.words('english')

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
ham1 = init_lists('.../enron1/ham/')
#spam2 = not used
#ham2 = not used
spam3 = init_lists("..../enron3/spam/")
ham3 = init_lists('..../enron3/ham/')
ham4 = init_lists("..../enron4/ham/")
ham6 = init_lists('..../enron6/ham/')

# combine two lists keeping the labels. 
# create a list of two-place tuples – Python objects, where the first member of the 
# pair stores the text of the email and the second one – its label
all_emails = [(email, 'spam') for email in spam1]
all_emails += [(email, 'ham') for email in ham1]
all_emails += [(email, 'spam') for email in spam3]
all_emails += [(email, 'ham') for email in ham3]
all_emails += [(email, 'ham') for email in ham4]
all_emails += [(email, 'ham') for email in ham6]               

# simple check that number of emails loaded into inital lists and final labelled list is the same
sum_ham = float(len(ham1)) + float(len(ham3)) + float(len(ham4)) + float(len(ham6))
print(sum_ham)
sum_spam = float(len(spam1)) + float(len(spam3))
print(sum_spam)              
print(sum_ham + sum_spam)               
print (len(all_emails))
# clear
del spam1[:]
del ham1[:]
del spam3[:]
del ham3[:]
del ham4[:]
del ham6[:]

# to randomly shuffle 'all_emails' the same way every time
random.shuffle(all_emails)

# PRE-PROCESS DATA for mulitnomial NB and SVC classifiers
# breaks email into individual words, removes words on stoplist, lemmatizes words 
# to word stem removes "words" shorter than 2 letters i.e. !"£$% 

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(unicode(sentence, errors='ignore'))
    return [lemmatizer.lemmatize(word.lower()) for word in words if len(word) > 2]

# Applies Normalised Term Frequency Representation to emails - Normalises text frequency 
# based on max occuring word in email                
def Normalise(text):
    text = [word for word in text if word not in stoplist]
    cnt = (Counter(text))
    maxword = (max(cnt.iteritems(), key = operator.itemgetter(1))[1])
    if sum(cnt.values()) > 0:
        factor=1.0/maxword
        cnt = {k: v*factor for k, v in cnt.items()}
        return cnt
    else:
        return cnt
            
def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    if setting== 'nbow':
        return {word: count for word, count in Normalise(preprocess(text)).items()}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

# PRE-PROCESS DATA "2" for Bernoulli NB Classifier
# breaks email into individual words, removes words on stoplist, lemmatizes words 
# to word stem removes "words" shorter than 2 letters i.e. !"£$% 
        
def preprocess2(sentence):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(unicode(sentence, errors='ignore'))
    lemma = [lemmatizer.lemmatize(word.lower()) for word in words if len(word) > 2]
    s=" "
    return s.join(wordl for wordl in lemma)

# structure email data in a format that works for the vectorizer - vectorizer is used to create 
# a 'bag-of-words" used by the  Bernoulli NB classifier
text = []
for i,j in all_emails:
    text.append(preprocess2(i))
   
labels = []
for i,j in all_emails:
    labels.append(unicode(j, encoding = 'ISO-8859-1', errors='ignore'))
    
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html     
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None,
                             stop_words = 'english',
                             lowercase = True,
                             max_features = 25000,
                             binary = True)

# CREATE FEATURE SET FOR BERNOULLI NAIVE BAYES CLASSIFIER
b_all_features = vectorizer.fit_transform(text)                                

# CREATE FEATURE SET FOR MULTINOMIAL AND SVC CLASSIFIERS
all_features = [(get_features(email, 'nbow'), label) for (email, label) in all_emails]
                                 
# Split dataset into Training and Test Sets for multinomial and SVC classifiers, default is for stratified
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
train_set, test_set = train_test_split(all_features, test_size = 0.2)

# # Split dataset into Training and Test Sets for bernoulli classifier, default is for stratified
b_train_set, b_test_set = train_test_split(b_all_features, test_size = 0.2)
Y_train_set, Y_test_set = train_test_split(labels, test_size = 0.2)

# TRAIN THE CLASSIFIERS
def train(features):
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')
    B_NB_classifier = BernoulliNB(alpha = 1, fit_prior = True)
    B_NB_classifier.fit(b_train_set, Y_train_set)
    
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(train_set)

    SVC_classifier = SklearnClassifier(SVC(C = 1, kernel = 'linear', cache_size = 500))
    SVC_classifier.train(train_set)

    LinearSVC_classifier = SklearnClassifier(LinearSVC(fit_intercept = False))
    LinearSVC_classifier.train(train_set)

    NuSVC_classifier = SklearnClassifier(NuSVC(nu=0.05, kernel='linear', cache_size=500))
    NuSVC_classifier.train(train_set)
    
    return B_NB_classifier, MNB_classifier, SVC_classifier, LinearSVC_classifier, NuSVC_classifier

# EVALUATION METHOD    
def evaluate(train_set, test_set, B_NB_classifier, MNB_classifier, SVC_classifier, LinearSVC_classifier, NuSVC_classifier):   
    
    # Confusion Matrix resources
    # http://www.nltk.org/_modules/nltk/classify/util.html#accuracy
    # http://www.nltk.org/book/ch06.html Ss 3.4 Confusion Matrix
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    # http://www.nltk.org/_modules/nltk/metrics/confusionmatrix.html#ConfusionMatrix    
    #  "reference" is the labelled test set its needed for the "right answers" 
    # in the confusion matrix and "results" the test_set results
    
    reference =[]
    for email, label in test_set:
        reference.append(label)
            
    # get a 2 item list of "spam" and "ham" for the confusion matrix
    class_names = list(set(reference))

    # ACA on test set
    B_NBresults = B_NB_classifier.predict(b_test_set)
    MNBresults = MNB_classifier.classify_many([fs for (fs, l) in test_set])
    SVCresults = SVC_classifier.classify_many([fs for (fs, l) in test_set])    
    LinSVCresults = LinearSVC_classifier.classify_many([fs for (fs, l) in test_set])    
    NuSVCresults = NuSVC_classifier.classify_many([fs for (fs, l) in test_set])
    
    # NBcm = nltk.ConfusionMatrix(reference, results)
    # print(NBcm.pretty_format(sort_by_count=True, show_percents=True))
    # based on http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
    # p 417/8 of Kelleher MacNamee & d'Arcy
    # reference (actual labels in test set), results (labels applied by classifier) (pos_label is a setting of recall_score)
    # Avg Class Acc uses the recall score for each class to be identified correctly
    B_NBposrecall = recall_score(Y_test_set, B_NBresults, pos_label = 'spam')
    B_NBnegrecall = recall_score(Y_test_set, B_NBresults, pos_label = 'ham')
    levels = float(len(class_names))
    average_class_accuracy = ((1/levels)*(B_NBposrecall + B_NBnegrecall))
    print ('B_NBC average class accuracy of the test set = ' + str(average_class_accuracy *100))

    MNBposrecall = recall_score(reference, MNBresults, pos_label = 'spam')
    MNBnegrecall = recall_score(reference, MNBresults, pos_label = 'ham')
    levels = float(len(class_names))
    average_class_accuracy = ((1/levels)*(MNBposrecall + MNBnegrecall))
    print ('MNB average class accuracy of the test set = ' + str(average_class_accuracy *100))
    
    SVCposrecall = recall_score(reference, SVCresults, pos_label = 'spam')
    SVCnegrecall = recall_score(reference, SVCresults, pos_label = 'ham')
    levels = float(len(class_names))
    average_class_accuracy = ((1/levels)*(SVCposrecall + SVCnegrecall))
    print ('SVC average class accuracy of the test set = ' + str(average_class_accuracy *100))

    LinSVCposrecall = recall_score(reference, LinSVCresults, pos_label = 'spam')
    LinSVCnegrecall = recall_score(reference, LinSVCresults, pos_label = 'ham')
    levels = float(len(class_names))
    average_class_accuracy = ((1/levels)*(LinSVCposrecall + LinSVCnegrecall))
    print ('Linear SVC average class accuracy of the test set = ' + str(average_class_accuracy *100))

    NuSVCposrecall = recall_score(reference, NuSVCresults, pos_label = 'spam')
    NuSVCnegrecall = recall_score(reference, NuSVCresults, pos_label = 'ham')
    levels = float(len(class_names))
    average_class_accuracy = ((1/levels)*(NuSVCposrecall + NuSVCnegrecall))
    print ('NuSVC average class accuracy of the test set = ' + str(average_class_accuracy *100))

# OUTPUT CONFUSION MATRICES FOR THE RESULTS OF THE CLASSIFIERS    
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        """if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)"""

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Compute confusion matrix for NB Classifier
    cnf_matrix = confusion_matrix(reference, B_NBresults)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix for NB Classifier
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix for Bernoulli Naive Bayes')   
    B_NBC_CNFM= plt.savefig('B_NBC_CNFM.png', bbox_inches='tight')
    plt.show()

    # Compute confusion matrix for MNB Classifier
    MNB_cnf_matrix = confusion_matrix(reference, MNBresults)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix for MNB Classifier
    plt.figure()
    plot_confusion_matrix(MNB_cnf_matrix, classes=class_names,
                      title='Confusion matrix Mutinomial NB')    
    MNB_CNFM= plt.savefig('MNB_CNFM.png', bbox_inches='tight')
    plt.show()
   
    # Compute confusion matrix for SVC Classifier using "Linear" kernal
    SVC_cnf_matrix = confusion_matrix(reference, SVCresults)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix for SVC Classifier using "Linear" kernal
    plt.figure()
    plot_confusion_matrix(SVC_cnf_matrix, classes=class_names,
                      title='Confusion matrix SVC w Linear Kernel')
    SVC_CNFM= plt.savefig('SVC_CNFM.png', bbox_inches='tight')
    plt.show()
    
    # Compute confusion matrix for Linear SVC Classifier
    LSVC_cnf_matrix = confusion_matrix(reference, LinSVCresults)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix for Liner Support Vector Machine Classifier
    plt.figure()
    plot_confusion_matrix(LSVC_cnf_matrix, classes=class_names,
                      title='Confusion matrix Linear SVC')
    LSVC_CNFM= plt.savefig('LSVC_CNFM.png', bbox_inches='tight')
    plt.show()
    
    # Compute confusion matrix for Linear SVC Classifier
    NuSVC_cnf_matrix = confusion_matrix(reference, NuSVCresults)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix for Liner Support Vector Machine Classifier
    plt.figure()
    plot_confusion_matrix(NuSVC_cnf_matrix, classes=class_names,
                      title='Confusion matrix NuSVC')
    NuSVC_CNFM= plt.savefig('NuSVC_CNFM.png', bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    # initialise the data
    print ('Corpus size = ' + str(len(all_emails)) + ' emails')
 
    # extract the features
    print ('Collected ' + str(len(all_features)) + ' feature sets')
 
    # train the classifier
    B_NB_classifier, MNB_classifier, SVC_classifier, LinearSVC_classifier, NuSVC_classifier = train(train_set)
 
    # evaluate its performance
    evaluate(train_set, test_set, B_NB_classifier, MNB_classifier, SVC_classifier , LinearSVC_classifier, NuSVC_classifier)
    