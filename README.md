# Automatic Diacritic Restoration of Yorùbá Text

### Motivations
Nigeria’s [dying languages](https://www.vanguardngr.com/2017/02/nigerias-dying-languages-warri-carry-last)

### Applications

* Generating very large, high quality Yorùbá text corpora
    * [physical books] → OCR → [undiacritized text] → ADR → [clean diacritized text]  Physical books written in Yorùbá (novels, manuals, school books, dictionaries) are digitized via [Optical Character Recognition](https://en.wikipedia.org/wiki/Optical_character_recognition) (OCR), which may not fully respect tonal or orthographic diacritics. Next, the undiacritized text is processed to restore the correct diacritics. 
    * Correcting digital texts scraped online on Twitter, Naija forums, articles, etc
    * Suggesting corrections during manual text entry (spell/diacritic checker)
    
* Preprocessing text for training Yorùbá
    * language models
    * word embeddings 
    * text-language identification (so Twitter can stop claiming Yorùbá text is Vietnamese haba!)
    * part-of-speech taggers
    * named-entity recognition
    * text-to-speech (TTS) models ([speech synthesis](https://keithito.com/LJ-Speech-Dataset/))
    * speech-to-text (STT) models ([speech recogntion](http://www.openslr.org/12/))

### Pretrained ADR Models
 * [New Pretrained Models (April 2020)](https://drive.google.com/drive/folders/1it32dyOHZWAeT7QDoTj-6qpo3b5addHT?usp=sharing) →  [Evaluation Results](https://github.com/Niger-Volta-LTI/yoruba-adr/blob/master/results/README.txt)
 * [Older Soft-attention models (March/June 2019)](https://bintray.com/ruohoruotsi/prebuilt-models/adr-models)

### Datasets
[https://github.com/Niger-Volta-LTI/yoruba-text](https://github.com/Niger-Volta-LTI/yoruba-text)

---
## Train a Yorùbá ADR model

### Dependencies
* Python3 (tested on 3.5, 3.6, 3.7)
* Install all dependencies: `pip3 install -r requirements.txt`

We train models on an Amazon EC2 `p2.xlarge` instance running `Deep Learning AMI (Ubuntu) Version 5.0 (ami-c27af5ba)`. These machine-images (AMI) have Python3 and PyTorch pre-installed as well as CUDA for training on the GPU. We use the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) framework for training and restoration.  

* To install PyTorch 0.4 manually, [follow instructions](https://pytorch.org/) for your {OS, package manager, python, CUDA} versions

* `git clone https://github.com/Niger-Volta-LTI/yoruba-adr.git`
* `git clone https://github.com/Niger-Volta-LTI/yoruba-text.git`

* Install dependencies: `pip3 install -r requirements.txt`
* Note that NLTK will need some extra hand-holding if you've installed it for the first time: 

	``` 
	Resource punkt not found.
  	Please use the NLTK Downloader to obtain the resource:

  	>>> import nltk
  	>>> nltk.download('punkt')
	```

#### Training an ADR sequence-to-sequence model
To start data-prep and training of the Bahdanau-style soft-attention model, execute the training script from the top-level  directory: `./01_run_training.sh` or `./01_run_training_transformer.sh`


## Learn more
* **Orthographic diacritics & multilingual computing**: http://www.phon.ucl.ac.uk/home/wells/dia/diacritics-revised.htm

* **Interspeech 2018 Paper**: [Attentive Sequence-to-Sequence Learning for Diacritic Restoration of Yorùbá Language Text](https://arxiv.org/abs/1804.00832)

* **ICLR 2020 AfricaNLP Workshop Paper**: [Improving Yorùbá Diacritic Restoration](https://arxiv.org/abs/2003.10564)


    If you use this code in your research please cite:
    ```
    @article{orife2018attentive,
      title={Attentive Sequence-to-Sequence Learning for Diacritic Restoration of Yor{\`u}B{\'a} Language Text},
      author={Orife, Iroro},
      journal={Proc. Interspeech 2018},
      pages={2848--2852},
      year={2018}
    }
    
    @article{orife2020improving,
      title={Improving Yor{\`u}B{\'a}  Diacritic Restoration},
      author={Orife, Iroro and Adelani, David I and Fasubaa, Timi and Williamson, Victor and Oyewusi, Wuraola Fisayo and Wahab, Olamilekan and Tubosun, Kola},
      journal={arXiv preprint arXiv:2003.10564},
      year={2020}
    }
    ```
