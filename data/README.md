### Dataset directory

This directory is populated during the course of running the training script. 

Source text is aggregated from various locations in [https://github.com/Niger-Volta-LTI/yoruba-text](https://github.com/Niger-Volta-LTI/yoruba-text) and split into train/dev/test sets. It is then processed to remove punctuation, select lines of max sequence length and remove diacritics to create a parallel source/target trainingset. 