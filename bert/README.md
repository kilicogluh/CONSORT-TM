# BioBERT-based CONSORT classification model

## Dependencies

The model requires __simpletransformers__ for execution. Original simpletransformers library is available at https://github.com/ThilinaRajapakse/simpletransformers. 

In order to create an environment to run the code, follow the steps below:
- create a conda/virtual environment called `transformers`
    - `conda create -n transformers python pandas tqdm`
    - `conda activate transformers`
- If using cuda:
    `conda install pytorch cudatoolkit=10.1 -c pytorch`  
    else: `conda install pytorch cpuonly -c pytorch`
- `pip install simpletransformers`
    


The directory structure of BERT is as follows
   
__models__ directory should be created and it should have required trained models to run the files


This corpus contains 50 randomized controlled trial articles annotated with 37 fine-grained [CONSORT checklist items](http://www.consort-statement.org/) at the sentence level. 

`data/50_XML` contains all the data in XML format. 

`bert` directory contains a [BioBERT](https://github.com/dmis-lab/biobert)-based model that labels Methods sentences with methodology-specific CONSORT items. Download the [model](https://drive.google.com/file/d/1FuLMQpIpsE9AEICqwm8BIU-ERB_jtZAt) and unzip it under `bert` directory to use it. This should create a directory named `bert/models`.

`svm` direcrtory contains a SVM classifier. 


## Models and Data

We used [BioBERT](https://github.com/dmis-lab/biobert) to train multi-label classification model on the CONSORT Methods items. The models should be downloaded from https://drive.google.com/file/d/1FuLMQpIpsE9AEICqwm8BIU-ERB_jtZAt/ and unzipped to create the `models` directory. Inside `models`, there are 3 subdirectories:
- __manual__ - Trained model on methods data without section information
- __manual_section__ - Model trained on methods data concatenated with section information
- __BioBert__ - pytorch format of __biobert__ used as a base model the experiments   
    
The training and test data are `data/Methods_train.csv` and `data/Methods_test.csv` files, respectively.

## Scripts

- __run__.py:  end-to-end script for executing the BioBERT-based model created for multi-label classification to be used on new data. It accepts input in the form a list of sentences for which __CONSORT__ labels are required. Currently it only supports __Methods__ related labels. 
- __train_bert_alone__.py - training a sentence-only model
     
     `cd CONSORT-TM`
     
    `python bert/train_bert_alone.py --model_save_path=models/model_name`
 
    Tunable/Command-Line Params
    - `data_path` - path to training data
    - `model_path` - path to BioBert model
    - `model_save_path` - path to save model
    
- __test_bert_alone__.py - testing for sentence-only model 
 
     `cd CONSORT-TM`
 
    `python bert/test_bert_alone.py`
    
    Tunable/Command-Line Params
    - `data_path` - path to testing data
    - `model_path` - path to saved trained model
    
 - __train_bert_section__.py - training with section names appended to sentence text
 
     `cd CONSORT-TM`
 
    `python bert/train_bert_section.py --model_save_path=models/model_name`
    
    Tunable/Command-Line Params
    - `data_path` - path to training data
    - `model_path` - path to BioBert model
    - `model_save_path` - path to save model
    
 -  __test_bert_section__.py - testing the model using the section information 
 
     `cd CONSORT-TM`
 
    `python bert/test_bert_section.py`
    
    Tunable/Command-Line Params
    - `data_path` - path to testing data
    - `model_path` - path to saved trained model
     
 

