# Automatic Diacritic Restoration of Yorùbá Text


### Motivations
Nigeria’s dying languages, [Warri carry last](https://www.vanguardngr.com/2017/02/nigerias-dying-languages-warri-carry-last)

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

### Dependencies
* Python3 (tested on 3.5, 3.6, 3.7)
* Install all dependencies: `pip3 install -r requirements.txt`


### Quickstart option #1
Correct diacritics with this [Jupyter notebook](https://github.com/Niger-Volta-LTI/yoruba-adr/blob/master/correct_yoruba_diacritics.ipynb)

### Quickstart option #2
Correct diacritics executing steps in the terminal

1) From the top-level directory, download a prebuilt model to the `./models` directory with `curl`:

    `$ curl -L "https://dl.bintray.com/ruohoruotsi/prebuilt-models/yo_adr_bahdanau_lstm_128_2_2_step_90000_release.pt" -o ./models/yo_adr_bahdanau_lstm_128_2_2_step_90000_release.pt`
2) Run the prediction script, which uses `./data/test/one_phrase.txt` and writes output file `./data/test/pred.txt`. Results are also written to terminal stdout:

    `$ ./02_run_prediction.sh`
    
    Terminal output should ressemble the following:
    ```
    [INFO] running inference on test sources
    [2019-04-18 00:44:08,754 INFO] Translating shard 0.
    SENT 1: ['awon', 'okunrin', 'nse', 'ise', 'agbara', 'bi', 'ise', 'ode']
    PRED 1: àwọn ọkùnrin nṣe iṣẹ́ agbára bí iṣẹ́ ọdẹ
    PRED SCORE: -0.8820
    PRED AVG SCORE: -0.1102, PRED PPL: 1.1166
    ```
    Verify the contents of the output file
    ```
    $ cat data/test/pred.txt
    àwọn ọkùnrin nṣe iṣẹ́ agbára bí iṣẹ́ ọdẹ
    ```
    
### Datasets
[https://github.com/Niger-Volta-LTI/yoruba-text](https://github.com/Niger-Volta-LTI/yoruba-text)


### Train a Yorùbá ADR model
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
To start data-prep and training of the Bahdanau-style soft-attention model, execute the training script from the top-level  directory: `./01_run_training.sh`

Training logging should ressemble the following:

```
[2018-09-03 17:28:27,544 INFO] Loading train dataset from data/demo.train.1.pt, number of examples: 42031
[2018-09-03 17:28:27,562 INFO]  * vocabulary size. source = 11090; target = 17720
[2018-09-03 17:28:27,562 INFO] Building model...
[2018-09-03 17:28:30,750 INFO] NMTModel(
  (encoder): RNNEncoder(
    (embeddings): Embeddings(
      (make_embedding): Sequential(
        (emb_luts): Elementwise(
          (0): Embedding(11090, 500, padding_idx=1)
        )
      )
    )
    (rnn): LSTM(500, 256, num_layers=2, dropout=0.3, bidirectional=True)
  )
  (decoder): InputFeedRNNDecoder(
    (embeddings): Embeddings(
      (make_embedding): Sequential(
        (emb_luts): Elementwise(
          (0): Embedding(17720, 500, padding_idx=1)
        )
      )
    )
    (dropout): Dropout(p=0.3)
    (rnn): StackedLSTM(
      (dropout): Dropout(p=0.3)
      (layers): ModuleList(
        (0): LSTMCell(1012, 512)
        (1): LSTMCell(512, 512)
      )
    )
    (attn): GlobalAttention(
      (linear_context): Linear(in_features=512, out_features=512, bias=False)
      (linear_query): Linear(in_features=512, out_features=512, bias=True)
      (v): Linear(in_features=512, out_features=1, bias=False)
      (linear_out): Linear(in_features=1024, out_features=512, bias=True)
      (sm): Softmax()
      (tanh): Tanh()
    )
  )
  (generator): Sequential(
    (0): Linear(in_features=512, out_features=17720, bias=True)
    (1): LogSoftmax()
  )
)
[2018-09-03 17:28:30,750 INFO] * number of parameters: 32901312
[2018-09-03 17:28:30,750 INFO] encoder: 8674344
[2018-09-03 17:28:30,750 INFO] decoder: 24226968
[2018-09-03 17:28:30,751 INFO] Making optimizer for training.
Stage 1: Keys after executing optim.set_parameters(model.parameters())
optim.optimizer.state_dict()['state'] keys: 
optim.optimizer.state_dict()['param_groups'] elements: 
optim.optimizer.state_dict()['param_groups'] element: {'lr': 1.0, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [140419757123696, 140419757123840, 140419757123984, 140419757124128, 140419757124272, 140419757124416, 140419757124560, 140419757124704, 140419757124848, 140419757125568, 140419757154448, 140419757154592, 140419757154736, 140419757154880, 140419757155024, 140419757155168, 140419757155312, 140419757156032, 140419757155888, 140419757156320, 140419757156464, 140419757156608, 140419757156752, 140419757156896, 140419757157040, 140419757157184, 140419757157328, 140419757157472, 140419757157616, 140419757157760, 140419757157904, 140419757158048, 140419757158192, 140419757158336]}
/home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/nn/functional.py:52: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))
[2018-09-03 17:28:30,752 INFO] 
[2018-09-03 17:28:30,752 INFO] Start training...
[2018-09-03 17:28:30,752 INFO]  * number of epochs: 31, starting from Epoch 1
[2018-09-03 17:28:30,752 INFO]  * batch size: 64
[2018-09-03 17:28:30,752 INFO] 
[2018-09-03 17:28:31,111 INFO] Loading train dataset from data/demo.train.1.pt, number of examples: 42031
[2018-09-03 17:28:38,838 INFO] Epoch  1,    50/  657; acc:   3.97; ppl: 124671.59; xent:  11.73;  6293 src tok/s; 6119 tgt tok/s;      8 s elapsed
[2018-09-03 17:28:46,608 INFO] Epoch  1,   100/  657; acc:   3.96; ppl: 18177.28; xent:   9.81;  6578 src tok/s; 6341 tgt tok/s;     15 s elapsed
[2018-09-03 17:28:54,283 INFO] Epoch  1,   150/  657; acc:   4.34; ppl: 4101.52; xent:   8.32;  6492 src tok/s; 6341 tgt tok/s;     23 s elapsed
[2018-09-03 17:29:01,799 INFO] Epoch  1,   200/  657; acc:   4.27; ppl: 3306.51; xent:   8.10;  6307 src tok/s; 6304 tgt tok/s;     31 s elapsed
[2018-09-03 17:29:09,630 INFO] Epoch  1,   250/  657; acc:   4.67; ppl: 1313.08; xent:   7.18;  6645 src tok/s; 6353 tgt tok/s;     39 s elapsed
[2018-09-03 17:29:16,987 INFO] Epoch  1,   300/  657; acc:   5.46; ppl: 1765.04; xent:   7.48;  6164 src tok/s; 6304 tgt tok/s;     46 s elapsed
[2018-09-03 17:29:24,582 INFO] Epoch  1,   350/  657; acc:   6.62; ppl: 1012.83; xent:   6.92;  6315 src tok/s; 6244 tgt tok/s;     53 s elapsed
[2018-09-03 17:29:32,078 INFO] Epoch  1,   400/  657; acc:   6.66; ppl: 956.28; xent:   6.86;  6332 src tok/s; 6343 tgt tok/s;     61 s elapsed
[2018-09-03 17:29:39,960 INFO] Epoch  1,   450/  657; acc:   7.15; ppl: 689.95; xent:   6.54;  6653 src tok/s; 6317 tgt tok/s;     69 s elapsed
[2018-09-03 17:29:47,263 INFO] Epoch  1,   500/  657; acc:   9.02; ppl: 830.68; xent:   6.72;  6097 src tok/s; 6286 tgt tok/s;     76 s elapsed
[2018-09-03 17:29:54,808 INFO] Epoch  1,   550/  657; acc:   9.15; ppl: 702.18; xent:   6.55;  6318 src tok/s; 6290 tgt tok/s;     84 s elapsed
[2018-09-03 17:30:02,537 INFO] Epoch  1,   600/  657; acc:  10.98; ppl: 545.61; xent:   6.30;  6511 src tok/s; 6313 tgt tok/s;     91 s elapsed
[2018-09-03 17:30:10,300 INFO] Epoch  1,   650/  657; acc:  12.64; ppl: 429.22; xent:   6.06;  6499 src tok/s; 6274 tgt tok/s;     99 s elapsed
[2018-09-03 17:30:11,223 INFO] Train perplexity: 1937.79
[2018-09-03 17:30:11,223 INFO] Train accuracy: 6.88972
[2018-09-03 17:30:11,257 INFO] Loading valid dataset from data/demo.valid.1.pt, number of examples: 5174
[2018-09-03 17:30:19,593 INFO] Validation perplexity: 507.708
[2018-09-03 17:30:19,594 INFO] Validation accuracy: 11.3796
```


### Training Performance

Ongoing best-results will be updated here and the corpus grows and training accuracy improves. Below are [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) summaries for the two models tested. 

##### Soft-Attention:
![alt text](https://github.com/Niger-Volta-LTI/yoruba-adr/blob/master/docs/soft-attention-tensorboard.png "WIP Soft-Attention training progress")

##### Self-Attention
![alt text](https://github.com/Niger-Volta-LTI/yoruba-adr/blob/master/docs/Transformer_training_loss.png "WIP Transformer training loss")

![alt text](https://github.com/Niger-Volta-LTI/yoruba-adr/blob/master/docs/Transformer_training_accuracy.png "WIP Transformer training accuracy")


## Learn more
* **Orthographic diacritics & multilingual computing**: http://www.phon.ucl.ac.uk/home/wells/dia/diacritics-revised.htm

* **Interspeech 2018 Paper**: [Attentive Sequence-to-Sequence Learning for Diacritic Restoration of Yorùbá Language Text](https://arxiv.org/abs/1804.00832)

    **Abstract** &rarr;
    _Yorùbá is a widely spoken West African language with a writing system rich in tonal and orthographic diacritics. With very few exceptions, diacritics are omitted from electronic texts, due to limited device and application support. Diacritics provide morphological information, are crucial for lexical disambiguation, pronunciation and are vital for any Yorùbá text-to-speech (TTS), automatic speech recognition (ASR) and natural language processing (NLP) tasks. Reframing Automatic Diacritic Restoration (ADR) as a machine translation task, we experiment with two different attentive Sequence-to-Sequence neural models to process undiacritized text. We have released pre-trained models, datasets and source-code as an open-source project to advance efforts on Yorùbá language technology._

    If you use this code in your research please cite:
    ```
    @article{orife2018attentive,
      title={Attentive Sequence-to-Sequence Learning for Diacritic Restoration of Yor{\`u}B{\'a} Language Text},
      author={Orife, Iroro},
      journal={Proc. Interspeech 2018},
      pages={2848--2852},
      year={2018}
    }
    ```
