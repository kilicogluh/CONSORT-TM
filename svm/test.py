#!/usr/bin/env python
# coding: utf-8

# In[14]:


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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC

import pathlib
current_folder = pathlib.Path().absolute()


def extract_feature(test_data_file, vocab_file):
    # Load testing data
    test_data_df = pd.read_csv(test_data_file)

    # Extract features
    tfidf_file = pickle.load(open(vocab_file, 'rb'))
    test_text = test_data_df['section'] + " " + test_data_df['text']
    test_text_features = tfidf_file.transform(test_text.values.astype('U'))

    # Define testing features and labels
    X_test = test_text_features
    y_test_dm = test_data_df['CONSORT_Item']

    # Prepare labels
    y_test = []
    for item in y_test_dm:
        item = item.replace("[","")
        item = item.replace("]","")
        item = item.replace("'","")
        item = item.replace(" ","")
        item = item.split(",")
        y_test.append(item)

    labels = ['10','11a','11b','12a','12b','3a','3b','4a','4b','5','6a','6b','7a','7b','8a','8b','9']
    mlb_labels = MultiLabelBinarizer(classes=labels)
    y_test = mlb_labels.fit_transform(y_test)
    
    return X_test, y_test,labels
   
def eval_model(model_file, X_test, y_test,labels):
    # Get testing results
    print ("Run model on test set...")
    loaded_model = pickle.load(open(model_file, 'rb'))
    clf = loaded_model['clf']
    binarizer = loaded_model['binarizer']
    
    predictions_multiclass = clf.predict(X_test)

    #Print Accurancy, ROC AUC, F1 Scores, Recall, Precision)
    print ("EVALUATION RESULTS:")
    print ('Accuracy:', accuracy_score(y_test, predictions_multiclass))
    print ('Precision:', precision_score(y_test, predictions_multiclass,average='micro'))
    print ('Recall:', recall_score(y_test, predictions_multiclass,average='micro'))
    print ('F1 score:', f1_score(y_test, predictions_multiclass,average='micro'))
    print ("ROC_AUC score:", roc_auc_score(y_test, predictions_multiclass))
    print (classification_report(y_test,predictions_multiclass, target_names=labels))


if __name__ == "__main__":
    current_folder = pathlib.Path().absolute()
    test_data_file = "../data/Methods_test.csv"
    vocab_file = str(current_folder) + "/models/vectorizer.pkl"
    model_file = str(current_folder) + "/models/model.sav"
    
    parser = argparse.ArgumentParser(description='CONSORT Classifier Evaluation.')
    parser.add_argument("--eval_data", default= test_data_file,required=False)
    args = parser.parse_args()
    print("Evaluation data file: ",args.eval_data)
    
    X_test, y_test,labels = extract_feature(args.eval_data,vocab_file)
    eval_model(model_file, X_test, y_test,labels)




