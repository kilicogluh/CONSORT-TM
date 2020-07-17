# BioBERT-based CONSORT classification model

## Dependencies

The models require __simpletransformers__ library, available at https://github.com/ThilinaRajapakse/simpletransformers. 

In order to create an environment to run the code, follow the steps below:
- create a conda/virtual environment called `transformers`
    - `conda create -n transformers python pandas tqdm`
    - `conda activate transformers`
- If using cuda:
    `conda install pytorch cudatoolkit=10.1 -c pytorch`  
    else: `conda install pytorch cpuonly -c pytorch`
- `pip install simpletransformers`
    

## Data 

The training and test data are in `data/Methods_train.csv` and `data/Methods_test.csv` files, respectively.

## Models 

Several [BioBERT](https://github.com/dmis-lab/biobert)-based multi-label classification models were generated. The appropriate model can be downloaded from the [model directory](https://drive.google.com/drive/folders/1Cx52lbcuuJ3SnwU9HVgXeBsyJY8g3rEG). Four models are available:
- [Text-only](https://drive.google.com/file/d/1jZI_I1aNnxUd4atSFz4PwOKqDJK16co4): the input to the model is text of the sentence to be classified only. 
- [Section-text](https://drive.google.com/file/d/1gj1NPz-gvf0WO21cxnF4GSiC-z552tt_): the input is the sentence text prepended with section header. This is the model used for the results reported in the paper. 
- [Text-only-50](https://drive.google.com/file/d/13dZkzbRqoxQfIY__pGunSzYNnGePIBVo): same as text-only, except trained on the entire dataset (50 articles). Appropriate for inference on new RCT articles, for which section information may not be available. 
- [Section-text-50](https://drive.google.com/file/d/1XYBwM6Q9nFmPTKdpPYsf5NuqpMLYSNZZ): same as section-text, trained on 50 articles. Appropriate when section information is available. 

Copy and unzip the model to __models__ directory. 

## Scripts

- __predict__.py:  end-to-end script for executing the BioBERT-based model created for multi-label classification to be used on new data. It accepts input in the form a list of sentences (one per line). It currently only supports __Methods__ related labels. 
- __train_bert__.py - training the sentence-only model
     
     `cd CONSORT-TM`
     
    `python bert/train_bert.py --model_save_path=models/model_name`
 
    Command-Line Parameters
    - `data_path` - path to training data
    - `model_path` - path to BioBert model
    - `model_save_path` - path to save model
    
- __test_bert__.py - testing for sentence-only model 
 
     `cd CONSORT-TM`
 
    `python bert/test_bert.py`
    
    Command-Line Parameters
    - `data_path` - path to testing data
    - `model_path` - path to saved trained model
    
 - __train_bert_section__.py - training with section names prepended to sentence text
 
     `cd CONSORT-TM`
 
    `python bert/train_bert_section.py --model_save_path=models/model_name`
    
    Command-Line Parameters
    - `data_path` - path to training data
    - `model_path` - path to BioBert model
    - `model_save_path` - path to save model
    
 -  __test_bert_section__.py - testing the model using the section information 
 
     `cd CONSORT-TM`
 
    `python bert/test_bert_section.py`
    
    Command-Line Parameters
    - `data_path` - path to testing data
    - `model_path` - path to saved trained model
     
 

