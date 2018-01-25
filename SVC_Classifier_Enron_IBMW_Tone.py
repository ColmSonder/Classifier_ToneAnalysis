# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:50:19 2016

@author: HP-Colm
"""

# THIS PART OF THE PROJECT TAKES THE DISCRETIZED SENTIMENT AND TONE FEATURES THAT WERE RETURNED FROM
# IBM WATSON TONE API, RUNS FEATURE SELECTION ON THE TONE FEATURES AND USING THE REDUCED DATASET APPLIES
# A SUPPORT VECTOR MACHINE CLASSIFIER TO CLASSIFY THE EMAILS AS SPAM OR HAM BASED ON THEIR SENTIMENT AND 
# TONE SCORES

# RESOURCES & GUIDES

# http://scikit-learn.org/stable/modules/feature_selection.html
# http://machinelearningmastery.com/feature-selection-machine-learning-python/
# http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
# http://machinelearningmastery.com/feature-selection-machine-learning-python/
# http://stackoverflow.com/questions/41475539/using-best-params-from-gridsearchcv

# IMPORT PACKAGES
#import random
from sklearn.svm import SVC, LinearSVC, NuSVC
import pandas as pd
from sklearn.model_selection import GridSearchCV
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

# Feature Selection with Univariate Statistical Tests (Chi-squared for classification)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif


enron_dataset_cat = pd.read_excel('...filepath to dataset.xlsx',index_col=0)

#print "enron dataset", len(enron_dataset)
print "enron dataset discretized", len(enron_dataset_cat)

# DATA PREP

#RANDOM SHUFFLE THE DATAFRAME
enron_dataset_cat = enron_dataset_cat.sample(frac=1).reset_index(drop=True)

#function to split (featureset,label) into (featureset) and (label)

def make_np_array_XY(xy):
    #print "make_np_array_XY()"
    a = np.array(xy)
    x = a[:,1:]
    y = a[:,0]
    return x,y

# split discrerized enron dataset into train, val and test set
XY_train_c, XY_test_c = np.split(enron_dataset_cat.sample(frac=1), [int(.8*len(enron_dataset_cat))])

# make featureset, label arrays for feature selection routines
X_train_c, Y_train_c = make_np_array_XY(XY_train_c)

# Feature Extraction with Univariate Statistical Tests (Mutual Information for classification)
# feature extraction
test = SelectKBest(score_func=chi2, k=13)
fit = test.fit(X_train_c, Y_train_c)

# summarize scores
np.set_printoptions(precision=3)
fit_scores = list(fit.scores_)
print(fit_scores)

#print("Num Features: %d") % fit.n_features_
features = fit.transform(X_train_c)
print(features.shape)
print("Selected Features: %s") % fit.get_support()


headers = list(XY_train_c)
headers.remove('label')
# if useing RFE then use below
# feat_select_bool = fit.support_

# if using SelectKBest use below
feat_select_bool = fit.get_support()

# REVISE TRAINING AND TEST SET TO ONLY CONTAIN THE BEST FEATURES AS DEFINED BY FEATURE SELECTION ROUTINE

feat_select_bool.tolist()
feat_select = zip(headers, feat_select_bool)
print(feat_select)
names_feat_select = []
names_feat_dropped = []
for i,j in feat_select:
   if j == True:
       names_feat_select.append(str(i))
for i,j in feat_select:
   if j == False:
       names_feat_dropped.append(str(i))

r_XY_train_c = XY_train_c.drop(names_feat_dropped, axis = 1)      
r_XY_test_c = XY_test_c.drop(names_feat_dropped, axis = 1)

X_train, Y_train = make_np_array_XY(r_XY_train_c)
X_test, Y_test= make_np_array_XY(r_XY_test_c)

# PARAMETER GRID CAN BE USED TO TRAIN ON A RANGE OF PARAMETER VALUES WITH THE BEST RESULT BEING OUTPUT
# THEN THE BEST SET CAN BE USED ON THE TEST SET

"""param_grid = [{'C': [0.1, 1, 10, 100], 'cache_size' : [500], 'kernel': ['linear']},
              {'C': [0.1, 1, 10, 100], 'cache_size' : [500],'gamma': [1e-1, 1, 1e1], 'kernel': ['rbf']},
              {'C': [0.1, 1, 10, 100], 'cache_size' : [500],'gamma': [1e-1, 1, 1e1], 'kernel': ['sigmoid']},]"""

## USING THE BEST PARAMETER SETTING FROM TRAINING APPLY TO TEST SET
param_grid = [{'C': [1], 'cache_size' : [500],'gamma': [10], 'kernel': ['rbf']}]
svr = SVC()
clf = GridSearchCV(svr, param_grid=param_grid, cv=6)
clf.fit(X_train, Y_train)
print(clf.best_params_)
svc_best_param_predict_test = clf.predict(X_test)

class_names = list(set(Y_train))

# CALCULATE AVERAGE CLASS ACCURACY OF SUPPORT VECTOR MACHINE CLASSIFIER
svc_bestposrecall = recall_score(Y_test, svc_best_param_predict_test, pos_label = 'spam')
svc_bestnegrecall = recall_score(Y_test, svc_best_param_predict_test, pos_label = 'ham')
levels = float(len(class_names))
svc1_average_class_accuracy = ((1/levels)*(svc_bestposrecall + svc_bestnegrecall))
print ('SVC average class accuracy of the test set = ' + str(svc1_average_class_accuracy *100))

# PLOT CONFUSION MATRIX FOR RESULTS
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

# Compute confusion matrix for SVC Classifier
svc_best_cnf_matrix = confusion_matrix(Y_test, svc_best_param_predict_test)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix for NB Classifier
plt.figure()
plot_confusion_matrix(svc_best_cnf_matrix, classes=class_names, title='Confusion matrix for SVC')

SVC_BEST_CNFM= plt.savefig('SVC_Tone_k13.png', bbox_inches='tight')
plt.show()
