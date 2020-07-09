### CONSORT Classifier

#### Prerequisites:
- python3
- nltk
- sklearn

#### Scripts:
- `train.py`: used to train the model
    * Parameters:
        - `train_data`: path to the training data (optional, default: `../data/Methods_train.csv`)
        - `override_vocab`: override existing vocab file (optional, default: True)  
        - `override_model`: override existing model file (optional, default: True) 
    * Output: 
        - Vocabulary file: `models/model.sav`
        - Model file: `models/vectorizer.pkl`
    * Example command lines:
        - To run default training: `python train.py`
        - To see the list of parameters: `python train.py -h`
        - To disable override vocab: `python --override_vocab False`
            
- `test.py`: used to test the model
    * Parameters:
        - `eval_data`: path to the evaluation data (optional, default: `../data/Methods_test.csv`)
    * Output: evaluation metrics, including Precision, Recall, F1-score and AUC ROC.
        
- `predict.py`: used to run the model on new data to get predictions
    * Parameters:
        - `predict_input_file`: path to the input data file for prediction. Data file should include columns "text" and "section", and one sentence per line.
        - `predict_output_file`: path to the output file to save the prediction results (optional, default: `prediction_results.csv`)
    * Example command line:
        - To run the model on a prediction data file: `python predict.py --predict_input_file predict_sample.csv`
