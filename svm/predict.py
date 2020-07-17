#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:


def extract_feature(prediction_data_file, vocab_file):
    # Load prediction data
    prediction_data_df = pd.read_csv(prediction_data_file)
    
    # Extract features
    tfidf_file = pickle.load(open(vocab_file, 'rb'))
    prediction_text = prediction_data_df['section'] + " " + prediction_data_df['sentence']
    prediction_text_features = tfidf_file.transform(prediction_text.values.astype('U'))

    # Define prediction features
    X_predict = prediction_text_features
    
    return X_predict



def predict(model_file, X_predict,prediction_data_file, output_file):
    loaded_model = pickle.load(open(model_file, 'rb'))
    clf = loaded_model['clf']
    binarizer = loaded_model['binarizer']
    
    predictions_multiclass = clf.predict(X_predict)
    predictions = binarizer.inverse_transform(predictions_multiclass)
    final_prediction = []
    for item in predictions:
        if len(item) == 0:
            preds = ["0"]
        else:
       	    preds = list(item)
        final_prediction.append(preds)
    
    prediction_data_df = pd.read_csv(prediction_data_file)
    prediction_data_df["predictions"] = final_prediction
    
    print ("Save prediction results into an output file...")
    prediction_data_df.to_csv(output_file, index = False)
    

if __name__ == "__main__":
    current_folder = pathlib.Path().absolute()
    vocab_file = str(current_folder) + "/models/vectorizer_50.pkl"
    model_file = str(current_folder) + "/models/model_50.sav"
    prediction_output_path = "prediction_results.csv"
    print (prediction_output_path)
    
    parser = argparse.ArgumentParser(description='CONSORT Classifier Prediction.')
    parser.add_argument("--predict_input_file")
    parser.add_argument("--predict_output_file", default=prediction_output_path, required = False)

    args = parser.parse_args()   
    
    print("Prediction data file: ",args.predict_input_file)
    
    X_predict = extract_feature(args.predict_input_file,vocab_file)
    predict(model_file, X_predict, args.predict_input_file,args.predict_output_file)

