CONSORT Classifier

Prerequisites:
    - Python 3
    - NLTK library
    - sklearn library

Scripts:
    - train.py: is used to train the model
        + Parameters:
            --train_data: path to the training data. Optional. Default: /data/manual_train_data_FINAL_18labels.csv
            --override_vocab: override existing vocab file. Optional. Default: True.  
            --override_model: override existing model file. Optional. Default: True. 
        + Output: 
            - Vocab file: /models/model.sav
            - Model file: /models/vectorizer.pkl
        + Example command lines:
            - To run default training: python train.py
            - To look at the list of parameters: python train.py -h
            - To disable override vocab: python --override_vocab False
        
             
    - test.py: is used to test the model
        + Parameters:
            --eval_data: path to the evaluation data. Optional. Default: /data/manual_test_data_FINAL_18labels.csv
        + Output: evaluation metrics, including Precision, Recall, F1-score and AUC ROC.
        
    - predict.py: is used to run the model on new data to get prediction
         + Parameters:
            --predict_input_file: path to the data file that you want to run model for prediction. Note that the data file should include two columns "text" and "section".
            --predict_output_file: path to the output file that you want to save the prediction results. Optional. Default: prediction_results.csv
         + Output: prediction_results.csv
         + Example command lines:
             - To run the model on a prediction data file: python predict.py --predict_input_file predict_sample.csv
        
        
        