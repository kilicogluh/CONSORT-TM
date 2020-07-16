## CONSORT-TM ##

This corpus contains 50 randomized controlled trial articles annotated with 37 fine-grained [CONSORT checklist items](http://www.consort-statement.org/) at the sentence level. 

`data/50_XML` contains all the data in XML format. 

`bert` directory contains a [BioBERT](https://github.com/dmis-lab/biobert)-based model that labels Methods sentences with methodology-specific CONSORT items. Download a zipped model file from the [model directory](https://drive.google.com/drive/folders/1Cx52lbcuuJ3SnwU9HVgXeBsyJY8g3rEG) and unzip it under `bert/models` directory to use the model.

`svm` directory contains SVM classifiers and relevant code.

## Contact

- Halil Kilicoglu:   (`halil (at) illinois.edu`) 
