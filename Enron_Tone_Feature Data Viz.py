# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:22:48 2016

@author: HP-Colm
"""
# RESOURCES GUIDES
# http://matplotlib.org/examples/pylab_examples/boxplot_demo.html
# http://stackoverflow.com/questions/33385238/how-to-convert-pandas-single-column-data-frame-to-series-or-numpy-vector
# https://bespokeblog.wordpress.com/2011/07/11/basic-data-plotting-with-matplotlib-part-3-histograms/ (not used)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load all dataframes
spam1 = pd.read_excel('..../enron1/enron1_dataset_w_tonefeats.xlsx')
ham1 = pd.read_excel('..../enron1/enron1_dataset_w_tonefeats1.xlsx')
spam3 = pd.read_excel('..../enron1/enron1_dataset_w_tonefeats2.xlsx')
ham3 = pd.read_excel('..../enron1/enron1_dataset_w_tonefeats3.xlsx')
ham4 = pd.read_excel('..../enron1/enron1_dataset_w_tonefeats4.xlsx')
ham6 = pd.read_excel('..../enron1/enron1_dataset_w_tonefeats5.xlsx')

# CHECK ON LOADS OF DATAFRAMES
sum_enron = float(len(spam1))+float(len(spam3))+float(len(ham1))+float(len(ham3))+float(len(ham4))+float(len(ham6))
print "spam1",  len(spam1)
print "spam3", len(spam3)
print "ham1", len(ham1)
print "ham3", len(ham3)
print "ham4", len(ham4)
print "ham6", len(ham6)
print "sum of dataframe lengths", sum_enron

# append all dataframes
enron_dataset = spam1
enron_dataset = enron_dataset.append(spam3, ignore_index=True)
enron_dataset = enron_dataset.append(ham1, ignore_index=True)
enron_dataset = enron_dataset.append(ham3, ignore_index=True)
enron_dataset = enron_dataset.append(ham4, ignore_index=True)
enron_dataset = enron_dataset.append(ham6, ignore_index=True)
print "enron dataset", len(enron_dataset)

# SPLIT DATAFRAME INTO HAM AND SPAM SETS TO COMPARE TONE FEATURE DISTRIBUTIONS
enron_spam_df = enron_dataset[enron_dataset['label'] == 'spam']
enron_ham_df = enron_dataset[enron_dataset['label'] == 'ham']

# CREATE HISTOGRAMS AND BOXPLOTS FOR EACH TONE FEATURE AND CLASS (SPAM & HAM)

#TONE ANGER
# create histograms and create boxplots
anger = enron_dataset.iloc[:,[1,2]]
spam_anger = anger[(anger.label == 'spam')]
ham_anger = anger[(anger.label == 'ham')]                

t_anger = anger['t_Anger'].values
s_anger = spam_anger['t_Anger'].values
h_anger = ham_anger['t_Anger'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_anger_bins, patches = plt.hist(t_anger, bins='auto')  
bin_mids = t_anger_bins[:-1] + np.diff(t_anger_bins)/2
enron_dataset['t_Anger_Cat'] = pd.cut(enron_dataset['t_Anger'], t_anger_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Anger")
plt.xlabel("Value")
plt.ylabel("Frequency")
SPAM_ASN_Anger_Hist = plt.savefig('Enron_Anger_Hist.png', bbox_inches='tight')
plt.show(block=False)

plt.figure()
plt.boxplot(t_anger, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Anger")
Enron_Anger_BoxP = plt.savefig('Enron_Anger_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_anger, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_anger, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham Anger Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_Anger_Hist = plt.savefig('Enron_Sp_Ham_Anger_Hist.png', bbox_inches='tight')
plt.show()

#TONE DISGUST
# create histograms and create boxplots
disgust = enron_dataset.iloc[:,[1,3]]
spam_disgust = disgust[(disgust.label == 'spam')]
ham_disgust = disgust[(disgust.label == 'ham')]                

t_disgust = disgust['t_Disgust'].values
s_disgust = spam_disgust['t_Disgust'].values
h_disgust = ham_disgust['t_Disgust'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_disgust_bins, patches = plt.hist(t_disgust, bins='auto')  
bin_mids = t_disgust_bins[:-1] + np.diff(t_disgust_bins)/2
enron_dataset['t_Disgust_Cat'] = pd.cut(enron_dataset['t_Disgust'], t_disgust_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Disgust")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_disgust_Hist = plt.savefig('Enron_disgust_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_disgust, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Disgust")
Enron_disgust_BoxP = plt.savefig('Enron_disgust_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_disgust, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_disgust, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham Disgust Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_disgust_Hist = plt.savefig('Enron_Sp_Ham_disgust_Hist.png', bbox_inches='tight')
plt.show()

#TONE FEAR
# create histograms and create boxplots
fear = enron_dataset.iloc[:,[1,4]]
spam_fear = fear[(fear.label == 'spam')]
ham_fear = fear[(fear.label == 'ham')]                

t_fear = fear['t_Fear'].values
s_fear = spam_fear['t_Fear'].values
h_fear = ham_fear['t_Fear'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_fear_bins, patches = plt.hist(t_fear, bins='auto')  
bin_mids = t_fear_bins[:-1] + np.diff(t_fear_bins)/2
enron_dataset['t_Fear_Cat'] = pd.cut(enron_dataset['t_Fear'], t_fear_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Fear")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_fear_Hist = plt.savefig('Enron_fear_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_fear, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Fear")
Enron_fear_BoxP = plt.savefig('Enron_fear_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_fear, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_fear, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham Fear Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_fear_Hist = plt.savefig('Enron_Sp_Ham_fear_Hist.png', bbox_inches='tight')
plt.show()

#TONE Joy
# create histograms and create boxplots
joy = enron_dataset.iloc[:,[1,5]]
spam_joy = joy[(joy.label == 'spam')]
ham_joy = joy[(joy.label == 'ham')]                

t_joy = joy['t_Joy'].values
s_joy = spam_joy['t_Joy'].values
h_joy = ham_joy['t_Joy'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_joy_bins, patches = plt.hist(t_joy, bins='auto')  
bin_mids = t_joy_bins[:-1] + np.diff(t_joy_bins)/2
enron_dataset['t_Joy_Cat'] = pd.cut(enron_dataset['t_Joy'], t_joy_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Joy")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_joy_Hist = plt.savefig('Enron_joy_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_joy, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Joy")
Enron_joy_BoxP = plt.savefig('Enron_joy_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_joy, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_joy, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham Joy Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_joy_Hist = plt.savefig('Enron_Sp_Ham_joy_Hist.png', bbox_inches='tight')
plt.show()

#TONE SADNESS
# create histograms and create boxplots
sadness = enron_dataset.iloc[:,[1,6]]
spam_sadness = sadness[(sadness.label == 'spam')]
ham_sadness = sadness[(sadness.label == 'ham')]                

t_sadness = sadness['t_Sadness'].values
s_sadness = spam_sadness['t_Sadness'].values
h_sadness = ham_sadness['t_Sadness'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_sadness_bins, patches = plt.hist(t_sadness, bins='auto')  
bin_mids = t_sadness_bins[:-1] + np.diff(t_sadness_bins)/2
enron_dataset['t_Sadness_Cat'] = pd.cut(enron_dataset['t_Sadness'], t_sadness_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Sadness")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_sadness_Hist = plt.savefig('Enron_sadness_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_sadness, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Sadness")
Enron_sadness_BoxP = plt.savefig('Enron_sadness_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_sadness, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_sadness, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham sadness Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_sadness_Hist = plt.savefig('Enron_Sp_Ham_Sadness_Hist.png', bbox_inches='tight')
plt.show()

#TONE ANALYTICAL
# create histograms and create boxplots
analytical = enron_dataset.iloc[:,[1,7]]
spam_analytical = analytical[(analytical.label == 'spam')]
ham_analytical = analytical[(analytical.label == 'ham')]                

t_analytical = analytical['t_Analytical'].values
s_analytical = spam_analytical['t_Analytical'].values
h_analytical = ham_analytical['t_Analytical'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_analytical_bins, patches = plt.hist(t_analytical, bins='auto')  
bin_mids = t_analytical_bins[:-1] + np.diff(t_analytical_bins)/2
enron_dataset['t_Analytical_Cat'] = pd.cut(enron_dataset['t_Analytical'], t_analytical_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Analytical")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_analytical_Hist = plt.savefig('Enron_analytical_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_analytical, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Analytical")
Enron_analytical_BoxP = plt.savefig('Enron_analytical_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_analytical, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_analytical, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham Analytical Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_analytical_Hist = plt.savefig('Enron_Sp_Ham_analytical_Hist.png', bbox_inches='tight')
plt.show()

#TONE CONFIDENT
# create histograms and create boxplots
confident = enron_dataset.iloc[:,[1,8]]
spam_confident = confident[(confident.label == 'spam')]
ham_confident = confident[(confident.label == 'ham')]                

t_confident = confident['t_Confident'].values
s_confident = spam_confident['t_Confident'].values
h_confident = ham_confident['t_Confident'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_confident_bins, patches = plt.hist(t_confident, bins=5)  
bin_mids = t_confident_bins[:-1] + np.diff(t_confident_bins)/2
enron_dataset['t_Confident_Cat'] = pd.cut(enron_dataset['t_Confident'], t_confident_bins, labels = bin_mids)
plt.title("Histogram with 5 bins for Confident")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_confident_Hist = plt.savefig('Enron_confident_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_confident, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Confident")
Enron_confident_BoxP = plt.savefig('Enron_confident_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_confident, bins=5, histtype='stepfilled', color='b', label='Spam')
plt.hist(h_confident, bins=5  , histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham Confident Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_confident_Hist = plt.savefig('Enron_Sp_Ham_confident_Hist.png', bbox_inches='tight')
plt.show()

#TONE TENTATIVE
# create histograms and create boxplots
tentative = enron_dataset.iloc[:,[1,9]]
spam_tentative = tentative[(tentative.label == 'spam')]
ham_tentative = tentative[(tentative.label == 'ham')]                

t_tentative = tentative['t_Tentative'].values
s_tentative = spam_tentative['t_Tentative'].values
h_tentative = ham_tentative['t_Tentative'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_tentative_bins, patches = plt.hist(t_tentative, bins='auto')  
bin_mids = t_tentative_bins[:-1] + np.diff(t_tentative_bins)/2
enron_dataset['t_Tentative_Cat'] = pd.cut(enron_dataset['t_Tentative'], t_tentative_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Tentative")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_tentative_Hist = plt.savefig('Enron_tentative_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_tentative, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Tentative")
Enron_tentative_BoxP = plt.savefig('Enron_tentative_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_tentative, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_tentative, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham Tentative Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_tentative_Hist = plt.savefig('Enron_Sp_Ham_tentative_Hist.png', bbox_inches='tight')

#TONE OPENNESS
# create histograms and create boxplots
openness = enron_dataset.iloc[:,[1,10]]
spam_openness = openness[(openness.label == 'spam')]
ham_openness = openness[(openness.label == 'ham')]                

t_openness = openness['t_Openness'].values
s_openness = spam_openness['t_Openness'].values
h_openness = ham_openness['t_Openness'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_openness_bins, patches = plt.hist(t_openness, bins='auto')  
bin_mids = t_openness_bins[:-1] + np.diff(t_openness_bins)/2
enron_dataset['t_Openness_Cat'] = pd.cut(enron_dataset['t_Openness'], t_openness_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Openness")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_openness_Hist = plt.savefig('Enron_openness_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_openness, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Openness")
Enron_openness_BoxP = plt.savefig('Enron_openness_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_openness, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_openness, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham Openness Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_openness_Hist = plt.savefig('Enron_Sp_Ham_openness_Hist.png', bbox_inches='tight')
plt.show()

#TONE CONSCIENTIOUSNESS
# create histograms and create boxplots
conscientiousness = enron_dataset.iloc[:,[1,11]]
spam_conscientiousness = conscientiousness[(conscientiousness.label == 'spam')]
ham_conscientiousness = conscientiousness[(conscientiousness.label == 'ham')]                

t_conscientiousness = conscientiousness['t_Conscientiousness'].values
s_conscientiousness = spam_conscientiousness['t_Conscientiousness'].values
h_conscientiousness = ham_conscientiousness['t_Conscientiousness'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_conscientiousness_bins, patches = plt.hist(t_conscientiousness, bins='auto')  
bin_mids = t_conscientiousness_bins[:-1] + np.diff(t_conscientiousness_bins)/2
enron_dataset['t_Conscientiousness_Cat'] = pd.cut(enron_dataset['t_Conscientiousness'], t_conscientiousness_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Conscientiousness")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_conscientiousness_Hist = plt.savefig('Enron_conscientiousness_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_conscientiousness, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Conscientiousness")
Enron_conscientiousness_BoxP = plt.savefig('Enron_conscientiousness_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_conscientiousness, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_conscientiousness, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham Conscientiousness Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_conscientiousness_Hist = plt.savefig('Enron_Sp_Ham_conscientiousness_Hist.png', bbox_inches='tight')
plt.show()

#TONE EXTRAVERSION
# create histograms and create boxplots
extraversion= enron_dataset.iloc[:,[1,12]]
spam_extraversion= extraversion[(extraversion.label == 'spam')]
ham_extraversion= extraversion[(extraversion.label == 'ham')]                

t_extraversion= extraversion['t_Extraversion'].values
s_extraversion= spam_extraversion['t_Extraversion'].values
h_extraversion= ham_extraversion['t_Extraversion'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_extraversion_bins, patches = plt.hist(t_extraversion, bins='auto')  
bin_mids = t_extraversion_bins[:-1] + np.diff(t_extraversion_bins)/2
enron_dataset['t_Extraversion_Cat'] = pd.cut(enron_dataset['t_Extraversion'], t_extraversion_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Extraversion")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_extraversion_Hist = plt.savefig('Enron_extraversion_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_extraversion, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Extraversion")
Enron_extraversion_BoxP = plt.savefig('Enron_extraversion_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_extraversion, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_extraversion, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham ExtraversionHistogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_extraversion_Hist = plt.savefig('Enron_Sp_Ham_extraversion_Hist.png', bbox_inches='tight')
plt.show()

#TONE AGREEABLENESS
# create histograms and create boxplots
agreeableness= enron_dataset.iloc[:,[1,13]]
spam_agreeableness= agreeableness[(agreeableness.label == 'spam')]
ham_agreeableness= agreeableness[(agreeableness.label == 'ham')]                

t_agreeableness= agreeableness['t_Agreeableness'].values
s_agreeableness= spam_agreeableness['t_Agreeableness'].values
h_agreeableness= ham_agreeableness['t_Agreeableness'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_agreeableness_bins, patches = plt.hist(t_agreeableness, bins='auto')  
bin_mids = t_agreeableness_bins[:-1] + np.diff(t_agreeableness_bins)/2
enron_dataset['t_Agreeableness_Cat'] = pd.cut(enron_dataset['t_Agreeableness'], t_agreeableness_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Agreeableness")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_agreeableness_Hist = plt.savefig('Enron_agreeableness_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_agreeableness, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Agreeableness")
Enron_agreeableness_BoxP = plt.savefig('Enron_agreeableness_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_agreeableness, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_agreeableness, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham Agreeableness Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_agreeableness_Hist = plt.savefig('Enron_Sp_Ham_agreeableness_Hist.png', bbox_inches='tight')
plt.show()

#TONE EMOTIONAL RANGE
# create histograms and create boxplots
emotionalrange= enron_dataset.iloc[:,[1,14]]
spam_emotionalrange= emotionalrange[(emotionalrange.label == 'spam')]
ham_emotionalrange= emotionalrange[(emotionalrange.label == 'ham')]                

t_emotionalrange= emotionalrange['t_EmotionalRange'].values
s_emotionalrange= spam_emotionalrange['t_EmotionalRange'].values
h_emotionalrange= ham_emotionalrange['t_EmotionalRange'].values

# plt.hist passes it's arguments to np.histogram, feature_bins saves the bin min/max values
# min_mids takes the mid-point of the bin min/max values so that it can be used as a category
# value to turn the continuous feature value into a categorical value for feature selection
               
n, t_emotionalrange_bins, patches = plt.hist(t_emotionalrange, bins='auto')  
bin_mids = t_emotionalrange_bins[:-1] + np.diff(t_emotionalrange_bins)/2
enron_dataset['t_EmotionalRange_Cat'] = pd.cut(enron_dataset['t_EmotionalRange'], t_emotionalrange_bins, labels = bin_mids)
plt.title("Histogram with 'auto' bins for Emotional Range")
plt.xlabel("Value")
plt.ylabel("Frequency")
Enron_emotionalrange_Hist = plt.savefig('Enron_emotionalrange_Hist.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.boxplot(t_emotionalrange, 0, 'rs', 0)
plt.title("BoxPlot with 'auto' bins for Emotional Range")
Enron_emotionalrange_BoxP = plt.savefig('Enron_emotionalrange_BoxP.png', bbox_inches='tight')
plt.show()

plt.hist(s_emotionalrange, bins='auto', histtype='stepfilled', color='b', label='Spam')
plt.hist(h_emotionalrange, bins='auto', histtype='stepfilled', color='r', alpha=0.5, label='Ham')
plt.title("Spam / Ham Emotional Range Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
Enron_Sp_Ham_emotionalrange_Hist = plt.savefig('Enron_Sp_Ham_emotionalrange_Hist.png', bbox_inches='tight')
plt.show()