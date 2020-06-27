#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os.path
import warnings
warnings.simplefilter("ignore", UserWarning)

import sys
import argparse

import pandas as pd
import pickle
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,roc_auc_score, roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import pathlib

# Read training data
def extract_feature(train_data_file, save_vocab):
    # Get training data
    train_data_df = pd.read_csv(train_data_file)

    # Extract features for training
    tfidf = TfidfVectorizer(stop_words=stop_words)
    train_text = train_data_df['section'] + " " + train_data_df['text']
    tfidf_vocab = tfidf.fit(train_text.values.astype('U'))
    train_text_features = tfidf_vocab.transform(train_text.values.astype('U'))

    # Define training features and labels
    X_train_dm = train_text_features
    y_train_dm = train_data_df['CONSORT_Item']
        
    if save_vocab == True: 
        vocab_file = str(current_folder) + "/models/vectorizer.pkl"
        with open(vocab_file, 'wb') as fin:
            pickle.dump(tfidf_vocab, fin)        

    X_train = X_train_dm

    # Prepare labels 
    y_train = []
    for item in y_train_dm:
        item = item.replace("[","")
        item = item.replace("]","")
        item = item.replace("'","")
        item = item.replace(" ","")
        item = item.split(",")
        y_train.append(item)

    labels = ['10','11a','11b','12a','12b','3a','3b','4a','4b','5','6a','6b','7a','7b','8a','8b','9']
    mlb_labels = MultiLabelBinarizer(classes=labels)
    y_train = mlb_labels.fit_transform(y_train)
    
    return X_train,y_train,mlb_labels


def train_model(X_train,y_train,save_model,mlb_labels):
    # Training 
    clf = OneVsRestClassifier(LinearSVC(C=10))

    # Fit the best algorithm to the data
    print ("Training...")
    clf_multiclass = clf.fit(X_train, y_train)
    if save_model == True:
        filename = str(current_folder) + '/models/model.sav'
        pickle.dump({'clf': clf_multiclass, 'binarizer': mlb_labels}, open(filename, 'wb'))

    print ("Done training!")


if __name__ == "__main__":
    current_folder = pathlib.Path().absolute()
    train_data_file = "../data/Methods_train.csv"
    
    parser = argparse.ArgumentParser(description='CONSORT Classifier Training. Please define parameters.')
    parser.add_argument("--train_data", default= train_data_file,required=False)
    parser.add_argument("--override_vocab", default = True,required=False)
    parser.add_argument("--override_model", default = True,required=False)
    args = parser.parse_args()
    print("Train data file: ",args.train_data)
    
    X_train, y_train,mlb_labels = extract_feature(args.train_data,args.override_vocab)
    train_model(X_train,y_train,args.override_model,mlb_labels)

        
