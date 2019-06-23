import numpy as np
import pandas as pd
import csv
from naive_bayes import naive_bayes
from sklearn.metrics import accuracy_score

#Introduction:

#For this example we are using the "forest dataset" which tries to predict
#the type of forest. You can download the dataset from the following link:
#https://www.kaggle.com/c/forest-cover-type-prediction


#This is a supervised learning project. We have to create an statistical model
#that maps a set of features into a label that we want to predict.

#Load the datasets. One dataset to training our classifiers and other dataset
#where we are going to use our ML model to predict
#We remove the first column as it does not provide extra information
with open("train.csv",'r') as forest_data:
    csv_reader=csv.DictReader(forest_data)
    attribute_names=csv_reader.fieldnames[1:]

forest_set=np.genfromtxt("train.csv", dtype=None, delimiter=',', skip_header=1)[:,1:]

#create a dictionary to store the result
results_dictionary={}

#Get information about our dataset
instances_dataset=len(forest_set)
attributes_dataset=forest_set.shape[1]-1

#Define the attributes and the labels we are predicting
attributes=forest_set[:,:attributes_dataset]
labels=forest_set[:,attributes_dataset]

#Get the number of unique labels
number_unique_labels=len(np.unique(labels))

#Split our dataset into a training and testing set
attributes_training, attributes_testing, labels_training, labels_testing=\
train_test_split(attributes,labels, test_size=0.20)

#Get the number of attributes
number_attributes=attributes_training.shape[1]


#In my implementation of naive we have to indicate the attributes we want to
#be consider as numerical or categorical. We create this matrix to do so.
numerical_indication_matrix=[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\
,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


#naive bayes implementation
#Get the class
naive_bayes_st=naive_bayes()

#classify
classification_matrix_naive_bayes=\
naive_bayes_st.naive_bayes_classifier(\
attributes_training,labels_training,attributes_testing,numerical_indication_matrix)

#Get the accuracy score of our naibe bayes classifier
nb_accuracy=accuracy_score(labels_testing,classification_matrix_naive_bayes)


#print the resut on the terminal
print nb_accuracy
