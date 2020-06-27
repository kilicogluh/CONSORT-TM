# CONSORT

CONsolidated Standards Of Reporting Trials aka CONSORT, is a set of guidelines for parallel group randomized controlled trials. There are 25 checklist items in the CONSORT statement
The aim of this repository is to classify each of the sentences in the critical trial journals.

## Notebooks
Notebooks folder contains mainly 3 important jupyter notebooks
1. __BERT__.ipynb - notebook to run BERT based models on the consort 

    __NOTES__ - Use this information while running BERT.ipynb
 - __use_top_section__ - Set this flag to __TRUE__ if you want to concat BERT embeddding with __embedding_column__ 
 - __embedding_column__ in BERT.ipynb is name of the column in train and test which would be used if __use_top_section__ is true. This should be in the form of a fixed dimension vector(we used __one hot__ vector in our case)
 
2. __data__.ipynb - convert manually annotated dataset from xml format to csv format with the following features
 - __PMCID__ - ID of the file/document from which sentence is taken
 - __sentence_id__ - unique identifier of the sentence
 - __sentence_text__ - text of the sentence
 - __start_char_pos__ - start position of the sentence in the entire document/file
 - __end_char_pos__ - end position of the sentence in the entire document/file
 - __section__ - section header of the sentence, e.g. __METHODS__
 - __CONSORT_Item__ - category/label to which sentence belongs
 - __top_section__ - highest order section header of the sentence
 
3. __xml_data_generator__.ipynb - extract __methods__ related data from raw __.ann__ files and convert them to csv format. Items extracted from these files include
 - __file_id__ - file from which the sentence is taken
 - __section__ - section of the sentence (related to methods section)
 - __sentence__ - text of the sentence


 ## Files 
 1. __BERT__.py - python version of the __BERT__.ipynb notebook for end-to-end execution for multi-label classification
 2. __run__.py - end to end execution of the BERT model created for multi-label classification to be used on new data. It accepts input in the form a list of sentences for which __CONSORT__ labels are required. Currently it only supports __Methods__ related labels 
 3. __bert_alone__.py - train and test model for just the sentences
 
    `python bert_alone.py --mode=test --data_path=/efs/CONSORT/MLDataset/manual_test_data_wMetaMap_FINAL_18labels.csv model_path=/efs/sahilw2/model/manual_linh_top_section_concat_2` 
 
     `python bert_alone.py --mode=train --data_path=/efs/CONSORT/MLDataset/manual_train_data_wMetaMap_FINAL_18labels.csv --model_path=/efs/sahilw2/model/BioBert/ --model_save_path=/efs/sahilw2/model/test_model` 
 4. __bert_w_section__.py - train and test model with sentences and section text
 
     `python bert_w_section.py --mode=test --data_path=/efs/CONSORT/MLDataset/manual_test_data_wMetaMap_FINAL_18labels.csv model_path=/efs/sahilw2/model/manual_linh_top_section_concat_2` 
 
     `python bert_w_section.py --mode=train --data_path=/efs/CONSORT/MLDataset/manual_train_data_wMetaMap_FINAL_18labels.csv --model_path=/efs/sahilw2/model/BioBert/ --model_save_path=/efs/sahilw2/model/test_model` 

## Models and Data

We used __BioBERT__ to train multi-label classification model on the CONSORT data

 - Data path - _/efs/sahilw2/data_
  - methods_data_new.csv - output of __xml_data_generator__.ipynb containing sentences. Input to __run__.py
  - manual_train_data_FINAL.csv and manual_test_data_FINAL.csv are methods section files used for training and testing  
 - Model path 
    - __/efs/sahilw2/model/manual_linh__ - Trained model on methods data without section information
    - __/efs/sahilw2/model/manual_linh_top_section_concat__ - Model trained on methods data concatenated with section information
    - __/efs/sahilw2/model/BioBert__ - Pytorch format of __biobert__ used as a base model for all the experiments in CONSORT    
 

## Dependencies

BERT model requires __simpletransformers__ for execution. Original simpletransformers library is available at https://github.com/ThilinaRajapakse/simpletransformers

However, I have made changes to the this library that enables us to append a fixed dimension embedding to the BERT 748 dimension output. In order to run BERT smoothly, install simpletransformers by following the following steps

- create a conda/virtual environment called `simpletransformers`
    - `conda create -n simpletransformers python pandas tqdm`
    - `conda activate simpletransformers`
- `cd simpletransformers`
- `python setup.py install`

