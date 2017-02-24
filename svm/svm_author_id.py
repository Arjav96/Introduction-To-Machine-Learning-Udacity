#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


from sklearn.svm import SVC
clf = SVC(C=10000.0,kernel='rbf')

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]


####time to train
t0 = time()

clf.fit(features_train,labels_train)

print "Training time: ",round(time()-t0,3),"s"

####time to predict
t1 = time()

pred = clf.predict(features_test)

print "Predict time: ",round(time()-t1,3),"s"


###print count of chris(or 1) predictions
import numpy as np  
print "Chris(1) predictions: ",np.count_nonzero(pred)

####print some predict results
"""print "test_element=10: ",pred[10]
print "test_element=26: ",pred[26]
print "test_element=50: ",pred[50]
"""


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)

print "accuracy: ",accuracy


#########################################################


