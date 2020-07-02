# CONSORT

CONsolidated Standards Of Reporting Trials aka CONSORT, is a set of guidelines for parallel group randomized controlled trials. There are 25 checklist items in the CONSORT statement
The aim of this repository is to classify each of the sentences in the critical trial journals.

The directory structure of BERT is as follows

+-- BERT
    
    +-- bert

    +-- data
    
    +-- models
   
__models__ directory should be created and it should have required trained models to run the files

## Models and Data

We used __BioBERT__ to train multi-label classification model on the CONSORT data. In order to test our models on your dataset, please download our models from the path https://drive.google.com/file/d/1FuLMQpIpsE9AEICqwm8BIU-ERB_jtZAt/ 

 - Inside __data__ path we have multiple files
  - manual_test_data_wMetaMap_FINAL_18labels.csv and manual_train_data_wMetaMap_FINAL_18labels.csv are methods section files used for testing and training respectively  
 - Inside __models__ directory, there are 3 folders 
    - __manual__ - Trained model on methods data without section information
    - __manual_section__ - Model trained on methods data concatenated with section information
    - __BioBert__ - Pytorch format of __biobert__ used as a base model for all the experiments in CONSORT   
    
## Files
 
 

Inside the directory __bert__, we have following files
 1. __run__.py - end to end execution of the BERT model created for multi-label classification to be used on new data. It accepts input in the form a list of sentences for which __CONSORT__ labels are required. Currently it only supports __Methods__ related labels 
 2. __train_bert_alone__.py - training file for just the sentences
     
     `cd BERT`
     
    `python bert/train_bert_alone.py --model_save_path=models/model_name`
 
    Tunable/Command-Line Params
    - data_path - path to training data
    - model_path - path to BioBert model
    - model_save_path - path to save model
 3.  __test_bert_alone__.py - test file for just the sentences
 
     `cd BERT`
 
    `python bert/test_bert_alone.py`
    
    Tunable/Command-Line Params
    - data_path - path to testing data
    - model_path - path to saved trained model
    
 4. __train_bert_section__.py - training file for section appended to sentence text
 
     `cd BERT`
 
    `python bert/train_bert_section.py --model_save_path=models/model_name`
    
    Tunable/Command-Line Params
    - data_path - path to training data
    - model_path - path to BioBert model
    - model_save_path - path to save model
 5.  __test_bert_section__.py - test file for just section appended to sentence text
 
     `cd BERT`
 
    `python bert/test_bert_section.py`
    
    Tunable/Command-Line Params
    - data_path - path to testing data
    - model_path - path to saved trained model
     
 

## Dependencies

BERT model requires __simpletransformers__ for execution. Original simpletransformers library is available at https://github.com/ThilinaRajapakse/simpletransformers

In order to create an environment to run our code follow the steps below - 

- create a conda/virtual environment called `transformers`
    - `conda create -n transformers python pandas tqdm`
    - `conda activate transformers`
- If using cuda:

    `conda install pytorch cudatoolkit=10.1 -c pytorch`
    
    else: `conda install pytorch cpuonly -c pytorch`
- `pip install simpletransformers`
    

