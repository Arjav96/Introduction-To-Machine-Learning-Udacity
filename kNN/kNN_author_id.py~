#!/usr/bin/python

""" 
    kNN Classifier to identify emails by their authors
    
    authors and labels:
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
from sklearn.neighbors import KNeighborsClassifier as kNN

clf = kNN(n_neighbors=3)

####time to train
t0 = time()

#train
clf.fit(features_train,labels_train)

print "training time:", round(time()-t0, 3), "s"


###time to predict
t1 = time()

#predict
pred = clf.predict(features_test)

print "testing time:", round(time()-t1, 3), "s"

#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)
print "accuracy: ",accuracy
#########################################################


