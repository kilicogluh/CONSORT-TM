## CONSORT-TM ##

This corpus contains 50 randomized controlled trial articles annotated with 37 fine-grained [CONSORT checklist items](http://www.consort-statement.org/) at the sentence level. 

`data/50_XML` contains all the data in XML format. 

`bert` directory contains a [BioBERT](https://github.com/dmis-lab/biobert)-based model that labels Methods sentences with methodology-specific CONSORT items. Download the [model](https://drive.google.com/file/d/1FuLMQpIpsE9AEICqwm8BIU-ERB_jtZAt) and unzip it under `bert` directory to use it. This should create a directory named `bert/models`.

`svm` direcrtory contains a SVM classifier. 

## Contact

- Halil Kilicoglu:   (`halil (at) illinois.edu`) 
