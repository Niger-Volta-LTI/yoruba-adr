# Automatic Diacritic Restoration of Yorùbá text


### Motivations
Nigeria’s dying languages, Warri carry last:
https://www.vanguardngr.com/2017/02/nigerias-dying-languages-warri-carry-last/

### Datasets
https://github.com/Niger-Volta-LTI/yoruba-text

### Orthographic diacritics and multilingual computing
http://www.phon.ucl.ac.uk/home/wells/dia/diacritics-revised.htm

### Applications

Very large Yorùbá text corpus generation (10M words) via the following process:
  
[physical books] → OCR → [undiacritized text] → ADR → [clean diacritized text]  

In this case, physical books written in Yorùbá (novels, manuals, school books, dictionaries) are digitized via [Optical Character Recognition](https://en.wikipedia.org/wiki/Optical_character_recognition) (OCR), which does not respect tonal or orthographic diacritics. Next, the undiacritized text is processed to restore the correct diacritics. 

Now we should have a large digital corpus with correct diacritics that can be used for training language models, word embeddings, fixing inputs to TTS programs, correcting text as well as seeding text for audio labeling  for asr (imagine Yorùbá [Librispeech](http://www.openslr.org/12/)


## Setup
We use [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) for training and translation.

* Use conda to install PyTorch, e.g.: `conda install pytorch torchvision -c pytorch` 
  * Alternatively, [follow instructions](https://pytorch.org/) for your {OS, package manager, python, CUDA}

* Install dependencies: `pip3 install -r requirements.txt`
