EMBEDDING TRANSFORMER RESULTS:

(pytorch_p36) ubuntu@ip-172-31-32-218:~/github/yoruba-adr$ more 01_run_training_transformer.sh
#!/usr/bin/env bash

#echo "[INFO] aggregate sources from yoruba-text, split & strip to make parallel text"
#python3 ./src/aggregate_corpora_make_parallel_text.py

#echo "[INFO] remove old tensorboard runs, and preprocessed files"
#rm data/*.pt
#rm -rf runs/*

#echo "[INFO] preprocess training data"
#python3 ./src/preprocess.py -train_src ./data/train/sources.txt \
#                        -train_tgt ./data/train/targets.txt \
#                        -valid_src ./data/dev/sources.txt \
#                        -valid_tgt ./data/dev/targets.txt \
#                        -save_data ./data/demo

echo "[INFO] running Transformer (self-attention) training, for GPU training add: -gpuid 0 "
# python3 ./src/train.py -gpuid 0 \
python3 ./src/train.py -world_size 1 -gpu_ranks 0 \
    -data data/demo \
    -save_model models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2 \
    -save_checkpoint_steps 500 \
    -tensorboard  \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 100000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 500 \
    -pre_word_vecs_enc "./embeddings.enc.pt" \
    -pre_word_vecs_dec "./embeddings.dec.pt"

(pytorch_p36) ubuntu@ip-172-31-32-218:~/github/yoruba-adr$ ./01_run_training_transformer.sh
[INFO] running Transformer (self-attention) training, for GPU training add: -gpuid 0
[2020-02-09 07:28:52,712 INFO]  * src vocab size = 43788
[2020-02-09 07:28:52,712 INFO]  * tgt vocab size = 50004
[2020-02-09 07:28:52,712 INFO] Building model...
[2020-02-09 07:29:01,567 INFO] NMTModel(
  (encoder): TransformerEncoder(
    (embeddings): Embeddings(
      (make_embedding): Sequential(
        (emb_luts): Elementwise(
          (0): Embedding(43788, 512, padding_idx=1)
        )
        (pe): PositionalEncoding(
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (transformer): ModuleList(
      (0): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (1): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (2): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (3): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (4): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (5): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
  )
  (decoder): TransformerDecoder(
    (embeddings): Embeddings(
      (make_embedding): Sequential(
        (emb_luts): Elementwise(
          (0): Embedding(50004, 512, padding_idx=1)
        )
        (pe): PositionalEncoding(
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (transformer_layers): ModuleList(
      (0): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm_1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (layer_norm_2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (drop): Dropout(p=0.1, inplace=False)
      )
      (1): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm_1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (layer_norm_2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (drop): Dropout(p=0.1, inplace=False)
      )
      (2): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm_1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (layer_norm_2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (drop): Dropout(p=0.1, inplace=False)
      )
      (3): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm_1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (layer_norm_2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (drop): Dropout(p=0.1, inplace=False)
      )
      (4): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm_1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (layer_norm_2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (drop): Dropout(p=0.1, inplace=False)
      )
      (5): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (softmax): Softmax(dim=-1)
          (dropout): Dropout(p=0.1, inplace=False)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (dropout_1): Dropout(p=0.1, inplace=False)
          (relu): ReLU()
          (dropout_2): Dropout(p=0.1, inplace=False)
        )
        (layer_norm_1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (layer_norm_2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (drop): Dropout(p=0.1, inplace=False)
      )
    )
    (layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
  )
  (generator): Sequential(
    (0): Linear(in_features=512, out_features=50004, bias=True)
    (1): Cast()
    (2): LogSoftmax()
  )
)
[2020-02-09 07:29:01,570 INFO] encoder: 41334784
[2020-02-09 07:29:01,570 INFO] decoder: 76479316
[2020-02-09 07:29:01,570 INFO] * number of parameters: 117814100
[2020-02-09 07:29:02,664 INFO] Starting training on GPU: [0]
[2020-02-09 07:29:02,664 INFO] Start training loop and validate every 500 steps...
[2020-02-09 07:29:02,665 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 07:29:09,291 INFO] number of examples: 368702
[2020-02-09 07:29:37,814 INFO] Step 50/100000; acc:   2.15; ppl: 11344.46; xent: 9.34; lr: 0.00001; 10559/11089 tok/s;     35 sec
[2020-02-09 07:30:02,326 INFO] Step 100/100000; acc:   4.41; ppl: 8355.35; xent: 9.03; lr: 0.00001; 15248/15953 tok/s;     60 sec
[2020-02-09 07:30:26,616 INFO] Step 150/100000; acc:   4.77; ppl: 4831.59; xent: 8.48; lr: 0.00002; 15262/16026 tok/s;     84 sec
[2020-02-09 07:30:51,072 INFO] Step 200/100000; acc:   4.54; ppl: 2257.74; xent: 7.72; lr: 0.00002; 15254/15980 tok/s;    108 sec
[2020-02-09 07:31:15,554 INFO] Step 250/100000; acc:   4.67; ppl: 987.24; xent: 6.89; lr: 0.00003; 15189/15933 tok/s;    133 sec
[2020-02-09 07:31:39,806 INFO] Step 300/100000; acc:   5.38; ppl: 482.94; xent: 6.18; lr: 0.00004; 15262/16049 tok/s;    157 sec
[2020-02-09 07:32:03,818 INFO] Step 350/100000; acc:   7.94; ppl: 306.45; xent: 5.73; lr: 0.00004; 15280/16148 tok/s;    181 sec
[2020-02-09 07:32:28,293 INFO] Step 400/100000; acc:   7.84; ppl: 249.44; xent: 5.52; lr: 0.00005; 15217/15953 tok/s;    206 sec
[2020-02-09 07:32:52,596 INFO] Step 450/100000; acc:   8.20; ppl: 228.23; xent: 5.43; lr: 0.00006; 15201/15954 tok/s;    230 sec
[2020-02-09 07:33:16,963 INFO] Step 500/100000; acc:  10.64; ppl: 204.53; xent: 5.32; lr: 0.00006; 15309/16036 tok/s;    254 sec
[2020-02-09 07:33:16,965 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 07:33:17,077 INFO] number of examples: 10264
[2020-02-09 07:33:27,536 INFO] Validation perplexity: 500.32
[2020-02-09 07:33:27,536 INFO] Validation accuracy: 12.0461
[2020-02-09 07:33:28,002 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_500.pt
[2020-02-09 07:33:54,521 INFO] Step 550/100000; acc:  15.07; ppl: 171.02; xent: 5.14; lr: 0.00007; 9924/10400 tok/s;    292 sec
[2020-02-09 07:34:19,124 INFO] Step 600/100000; acc:  18.85; ppl: 132.69; xent: 4.89; lr: 0.00007; 15169/15883 tok/s;    316 sec
[2020-02-09 07:34:43,521 INFO] Step 650/100000; acc:  23.31; ppl: 101.83; xent: 4.62; lr: 0.00008; 15263/16004 tok/s;    341 sec
[2020-02-09 07:35:08,085 INFO] Step 700/100000; acc:  29.83; ppl: 71.37; xent: 4.27; lr: 0.00009; 15238/15932 tok/s;    365 sec
[2020-02-09 07:35:32,649 INFO] Step 750/100000; acc:  35.42; ppl: 52.69; xent: 3.96; lr: 0.00009; 15130/15878 tok/s;    390 sec
[2020-02-09 07:35:56,904 INFO] Step 800/100000; acc:  41.37; ppl: 36.61; xent: 3.60; lr: 0.00010; 15197/16009 tok/s;    414 sec
[2020-02-09 07:36:20,945 INFO] Step 850/100000; acc:  47.45; ppl: 24.95; xent: 3.22; lr: 0.00011; 15297/16140 tok/s;    438 sec
[2020-02-09 07:36:45,408 INFO] Step 900/100000; acc:  49.21; ppl: 21.10; xent: 3.05; lr: 0.00011; 15173/15936 tok/s;    463 sec
[2020-02-09 07:37:09,772 INFO] Step 950/100000; acc:  55.87; ppl: 14.33; xent: 2.66; lr: 0.00012; 15279/16024 tok/s;    487 sec
[2020-02-09 07:37:34,349 INFO] Step 1000/100000; acc:  59.25; ppl: 11.19; xent: 2.41; lr: 0.00012; 15218/15916 tok/s;    512 sec
[2020-02-09 07:37:34,351 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 07:37:34,468 INFO] number of examples: 10264
[2020-02-09 07:37:44,772 INFO] Validation perplexity: 20.7909
[2020-02-09 07:37:44,773 INFO] Validation accuracy: 56.2016
[2020-02-09 07:37:45,241 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_1000.pt
[2020-02-09 07:37:49,808 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 07:37:58,166 INFO] number of examples: 368702
[2020-02-09 07:38:24,164 INFO] Step 1050/100000; acc:  61.94; ppl:  9.31; xent: 2.23; lr: 0.00013; 7481/7836 tok/s;    561 sec
[2020-02-09 07:38:48,647 INFO] Step 1100/100000; acc:  64.84; ppl:  7.75; xent: 2.05; lr: 0.00014; 15186/15933 tok/s;    586 sec
[2020-02-09 07:39:13,149 INFO] Step 1150/100000; acc:  68.53; ppl:  6.21; xent: 1.83; lr: 0.00014; 15185/15916 tok/s;    610 sec
[2020-02-09 07:39:37,642 INFO] Step 1200/100000; acc:  69.06; ppl:  5.79; xent: 1.76; lr: 0.00015; 15204/15942 tok/s;    635 sec
[2020-02-09 07:40:02,258 INFO] Step 1250/100000; acc:  71.35; ppl:  5.03; xent: 1.61; lr: 0.00015; 15152/15869 tok/s;    660 sec
[2020-02-09 07:40:26,630 INFO] Step 1300/100000; acc:  72.76; ppl:  4.58; xent: 1.52; lr: 0.00016; 15172/15965 tok/s;    684 sec
[2020-02-09 07:40:50,710 INFO] Step 1350/100000; acc:  75.30; ppl:  3.98; xent: 1.38; lr: 0.00017; 15246/16104 tok/s;    708 sec
[2020-02-09 07:41:15,112 INFO] Step 1400/100000; acc:  76.95; ppl:  3.58; xent: 1.27; lr: 0.00017; 15201/15969 tok/s;    732 sec
[2020-02-09 07:41:39,566 INFO] Step 1450/100000; acc:  77.61; ppl:  3.40; xent: 1.22; lr: 0.00018; 15153/15879 tok/s;    757 sec
[2020-02-09 07:42:03,851 INFO] Step 1500/100000; acc:  78.96; ppl:  3.12; xent: 1.14; lr: 0.00019; 15317/16069 tok/s;    781 sec
[2020-02-09 07:42:03,852 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 07:42:03,957 INFO] number of examples: 10264
[2020-02-09 07:42:14,240 INFO] Validation perplexity: 5.53963
[2020-02-09 07:42:14,240 INFO] Validation accuracy: 72.0324
[2020-02-09 07:42:14,701 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_1500.pt
[2020-02-09 07:42:41,304 INFO] Step 1550/100000; acc:  79.17; ppl:  3.03; xent: 1.11; lr: 0.00019; 9987/10447 tok/s;    819 sec
[2020-02-09 07:43:05,800 INFO] Step 1600/100000; acc:  81.04; ppl:  2.74; xent: 1.01; lr: 0.00020; 15193/15931 tok/s;    843 sec
[2020-02-09 07:43:30,342 INFO] Step 1650/100000; acc:  79.75; ppl:  2.89; xent: 1.06; lr: 0.00020; 15201/15925 tok/s;    868 sec
[2020-02-09 07:43:54,984 INFO] Step 1700/100000; acc:  82.20; ppl:  2.52; xent: 0.93; lr: 0.00021; 15204/15887 tok/s;    892 sec
[2020-02-09 07:44:19,522 INFO] Step 1750/100000; acc:  82.82; ppl:  2.45; xent: 0.89; lr: 0.00022; 15137/15888 tok/s;    917 sec
[2020-02-09 07:44:43,778 INFO] Step 1800/100000; acc:  83.28; ppl:  2.38; xent: 0.87; lr: 0.00022; 15200/16012 tok/s;    941 sec
[2020-02-09 07:45:07,862 INFO] Step 1850/100000; acc:  85.62; ppl:  2.11; xent: 0.74; lr: 0.00023; 15272/16119 tok/s;    965 sec
[2020-02-09 07:45:32,295 INFO] Step 1900/100000; acc:  84.17; ppl:  2.25; xent: 0.81; lr: 0.00023; 15159/15933 tok/s;    990 sec
[2020-02-09 07:45:56,702 INFO] Step 1950/100000; acc:  85.35; ppl:  2.11; xent: 0.75; lr: 0.00024; 15260/16000 tok/s;   1014 sec
[2020-02-09 07:46:21,267 INFO] Step 2000/100000; acc:  86.71; ppl:  1.96; xent: 0.67; lr: 0.00025; 15192/15907 tok/s;   1039 sec
[2020-02-09 07:46:21,269 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 07:46:21,373 INFO] number of examples: 10264
[2020-02-09 07:46:31,674 INFO] Validation perplexity: 3.18961
[2020-02-09 07:46:31,674 INFO] Validation accuracy: 79.887
[2020-02-09 07:46:32,140 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_2000.pt
[2020-02-09 07:46:39,201 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 07:46:45,020 INFO] number of examples: 368702
[2020-02-09 07:47:08,596 INFO] Step 2050/100000; acc:  86.66; ppl:  1.95; xent: 0.67; lr: 0.00025; 7899/8261 tok/s;   1086 sec
[2020-02-09 07:47:32,980 INFO] Step 2100/100000; acc:  87.24; ppl:  1.91; xent: 0.65; lr: 0.00026; 15206/15977 tok/s;   1110 sec
[2020-02-09 07:47:57,589 INFO] Step 2150/100000; acc:  87.55; ppl:  1.86; xent: 0.62; lr: 0.00027; 15161/15870 tok/s;   1135 sec
[2020-02-09 07:48:22,063 INFO] Step 2200/100000; acc:  88.05; ppl:  1.82; xent: 0.60; lr: 0.00027; 15197/15942 tok/s;   1159 sec
[2020-02-09 07:48:46,631 INFO] Step 2250/100000; acc:  88.65; ppl:  1.76; xent: 0.57; lr: 0.00028; 15168/15894 tok/s;   1184 sec
[2020-02-09 07:49:10,980 INFO] Step 2300/100000; acc:  88.51; ppl:  1.78; xent: 0.58; lr: 0.00028; 15176/15974 tok/s;   1208 sec
[2020-02-09 07:49:35,175 INFO] Step 2350/100000; acc:  88.37; ppl:  1.78; xent: 0.58; lr: 0.00029; 15240/16061 tok/s;   1233 sec
[2020-02-09 07:49:59,509 INFO] Step 2400/100000; acc:  89.57; ppl:  1.68; xent: 0.52; lr: 0.00030; 15193/15989 tok/s;   1257 sec
[2020-02-09 07:50:23,969 INFO] Step 2450/100000; acc:  90.19; ppl:  1.62; xent: 0.48; lr: 0.00030; 15172/15885 tok/s;   1281 sec
[2020-02-09 07:50:48,307 INFO] Step 2500/100000; acc:  90.12; ppl:  1.63; xent: 0.49; lr: 0.00031; 15273/16029 tok/s;   1306 sec
[2020-02-09 07:50:48,308 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 07:50:48,417 INFO] number of examples: 10264
[2020-02-09 07:50:58,748 INFO] Validation perplexity: 2.46713
[2020-02-09 07:50:58,749 INFO] Validation accuracy: 84.2
[2020-02-09 07:50:59,216 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_2500.pt
[2020-02-09 07:51:25,906 INFO] Step 2550/100000; acc:  90.02; ppl:  1.62; xent: 0.49; lr: 0.00032; 9967/10415 tok/s;   1343 sec
[2020-02-09 07:51:50,368 INFO] Step 2600/100000; acc:  90.90; ppl:  1.57; xent: 0.45; lr: 0.00032; 15190/15942 tok/s;   1368 sec
[2020-02-09 07:52:14,887 INFO] Step 2650/100000; acc:  90.98; ppl:  1.56; xent: 0.44; lr: 0.00033; 15216/15941 tok/s;   1392 sec
[2020-02-09 07:52:39,495 INFO] Step 2700/100000; acc:  91.18; ppl:  1.54; xent: 0.43; lr: 0.00033; 15219/15905 tok/s;   1417 sec
[2020-02-09 07:53:03,955 INFO] Step 2750/100000; acc:  90.29; ppl:  1.61; xent: 0.48; lr: 0.00034; 15172/15936 tok/s;   1441 sec
[2020-02-09 07:53:28,292 INFO] Step 2800/100000; acc:  90.46; ppl:  1.60; xent: 0.47; lr: 0.00035; 15148/15954 tok/s;   1466 sec
[2020-02-09 07:53:52,323 INFO] Step 2850/100000; acc:  91.71; ppl:  1.51; xent: 0.41; lr: 0.00035; 15296/16150 tok/s;   1490 sec
[2020-02-09 07:54:16,696 INFO] Step 2900/100000; acc:  91.77; ppl:  1.51; xent: 0.41; lr: 0.00036; 15195/15972 tok/s;   1514 sec
[2020-02-09 07:54:41,065 INFO] Step 2950/100000; acc:  92.38; ppl:  1.45; xent: 0.37; lr: 0.00036; 15277/16021 tok/s;   1538 sec
[2020-02-09 07:55:05,809 INFO] Step 3000/100000; acc:  92.38; ppl:  1.45; xent: 0.37; lr: 0.00037; 15148/15824 tok/s;   1563 sec
[2020-02-09 07:55:05,811 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 07:55:05,918 INFO] number of examples: 10264
[2020-02-09 07:55:16,279 INFO] Validation perplexity: 2.16238
[2020-02-09 07:55:16,279 INFO] Validation accuracy: 86.5449
[2020-02-09 07:55:16,746 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_3000.pt
[2020-02-09 07:55:26,169 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 07:55:33,152 INFO] number of examples: 368702
[2020-02-09 07:55:54,241 INFO] Step 3050/100000; acc:  92.82; ppl:  1.42; xent: 0.35; lr: 0.00038; 7715/8072 tok/s;   1612 sec
[2020-02-09 07:56:18,685 INFO] Step 3100/100000; acc:  92.58; ppl:  1.44; xent: 0.36; lr: 0.00038; 15166/15933 tok/s;   1636 sec
[2020-02-09 07:56:43,248 INFO] Step 3150/100000; acc:  92.99; ppl:  1.42; xent: 0.35; lr: 0.00039; 15164/15889 tok/s;   1661 sec
[2020-02-09 07:57:07,650 INFO] Step 3200/100000; acc:  93.36; ppl:  1.39; xent: 0.33; lr: 0.00040; 15226/15980 tok/s;   1685 sec
[2020-02-09 07:57:32,343 INFO] Step 3250/100000; acc:  93.12; ppl:  1.40; xent: 0.34; lr: 0.00040; 15147/15843 tok/s;   1710 sec
[2020-02-09 07:57:56,741 INFO] Step 3300/100000; acc:  89.43; ppl:  1.72; xent: 0.54; lr: 0.00041; 15117/15926 tok/s;   1734 sec
[2020-02-09 07:58:20,923 INFO] Step 3350/100000; acc:  92.33; ppl:  1.46; xent: 0.38; lr: 0.00041; 15236/16063 tok/s;   1758 sec
[2020-02-09 07:58:45,241 INFO] Step 3400/100000; acc:  93.44; ppl:  1.39; xent: 0.33; lr: 0.00042; 15191/15994 tok/s;   1783 sec
[2020-02-09 07:59:09,788 INFO] Step 3450/100000; acc:  93.77; ppl:  1.36; xent: 0.31; lr: 0.00043; 15198/15915 tok/s;   1807 sec
[2020-02-09 07:59:34,117 INFO] Step 3500/100000; acc:  93.52; ppl:  1.37; xent: 0.32; lr: 0.00043; 15205/15952 tok/s;   1831 sec
[2020-02-09 07:59:34,119 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 07:59:34,228 INFO] number of examples: 10264
[2020-02-09 07:59:44,579 INFO] Validation perplexity: 1.9858
[2020-02-09 07:59:44,579 INFO] Validation accuracy: 87.9138
[2020-02-09 07:59:45,042 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_3500.pt
[2020-02-09 08:00:11,702 INFO] Step 3550/100000; acc:  93.87; ppl:  1.35; xent: 0.30; lr: 0.00044; 9975/10420 tok/s;   1869 sec
[2020-02-09 08:00:36,062 INFO] Step 3600/100000; acc:  94.00; ppl:  1.34; xent: 0.30; lr: 0.00044; 15220/15992 tok/s;   1893 sec
[2020-02-09 08:01:00,703 INFO] Step 3650/100000; acc:  94.16; ppl:  1.33; xent: 0.29; lr: 0.00045; 15170/15877 tok/s;   1918 sec
[2020-02-09 08:01:25,267 INFO] Step 3700/100000; acc:  94.28; ppl:  1.32; xent: 0.28; lr: 0.00046; 15235/15928 tok/s;   1943 sec
[2020-02-09 08:01:49,748 INFO] Step 3750/100000; acc:  94.19; ppl:  1.33; xent: 0.29; lr: 0.00046; 15155/15919 tok/s;   1967 sec
[2020-02-09 08:02:14,156 INFO] Step 3800/100000; acc:  93.96; ppl:  1.35; xent: 0.30; lr: 0.00047; 15101/15908 tok/s;   1991 sec
[2020-02-09 08:02:38,344 INFO] Step 3850/100000; acc:  94.40; ppl:  1.32; xent: 0.27; lr: 0.00048; 15284/16088 tok/s;   2016 sec
[2020-02-09 08:03:02,661 INFO] Step 3900/100000; acc:  94.26; ppl:  1.33; xent: 0.28; lr: 0.00048; 15162/15976 tok/s;   2040 sec
[2020-02-09 08:03:27,070 INFO] Step 3950/100000; acc:  94.45; ppl:  1.31; xent: 0.27; lr: 0.00049; 15249/15994 tok/s;   2064 sec
[2020-02-09 08:03:51,770 INFO] Step 4000/100000; acc:  94.69; ppl:  1.30; xent: 0.26; lr: 0.00049; 15168/15849 tok/s;   2089 sec
[2020-02-09 08:03:51,772 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:03:52,934 INFO] number of examples: 10264
[2020-02-09 08:04:03,264 INFO] Validation perplexity: 1.97961
[2020-02-09 08:04:03,264 INFO] Validation accuracy: 88.0846
[2020-02-09 08:04:03,726 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_4000.pt
[2020-02-09 08:04:15,626 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 08:04:22,756 INFO] number of examples: 368702
[2020-02-09 08:04:41,328 INFO] Step 4050/100000; acc:  94.85; ppl:  1.28; xent: 0.25; lr: 0.00050; 7535/7887 tok/s;   2139 sec
[2020-02-09 08:05:05,740 INFO] Step 4100/100000; acc:  94.54; ppl:  1.31; xent: 0.27; lr: 0.00051; 15190/15955 tok/s;   2163 sec
[2020-02-09 08:05:30,308 INFO] Step 4150/100000; acc:  94.84; ppl:  1.29; xent: 0.25; lr: 0.00051; 15177/15894 tok/s;   2188 sec
[2020-02-09 08:05:54,746 INFO] Step 4200/100000; acc:  95.08; ppl:  1.27; xent: 0.24; lr: 0.00052; 15204/15956 tok/s;   2212 sec
[2020-02-09 08:06:19,371 INFO] Step 4250/100000; acc:  94.88; ppl:  1.28; xent: 0.25; lr: 0.00053; 15148/15866 tok/s;   2237 sec
[2020-02-09 08:06:43,834 INFO] Step 4300/100000; acc:  94.98; ppl:  1.28; xent: 0.25; lr: 0.00053; 15128/15909 tok/s;   2261 sec
[2020-02-09 08:07:08,036 INFO] Step 4350/100000; acc:  94.91; ppl:  1.28; xent: 0.25; lr: 0.00054; 15206/16042 tok/s;   2285 sec
[2020-02-09 08:07:32,473 INFO] Step 4400/100000; acc:  95.09; ppl:  1.27; xent: 0.24; lr: 0.00054; 15141/15927 tok/s;   2310 sec
[2020-02-09 08:07:57,028 INFO] Step 4450/100000; acc:  95.31; ppl:  1.26; xent: 0.23; lr: 0.00055; 15170/15899 tok/s;   2334 sec
[2020-02-09 08:08:21,420 INFO] Step 4500/100000; acc:  94.28; ppl:  1.32; xent: 0.28; lr: 0.00056; 15197/15925 tok/s;   2359 sec
[2020-02-09 08:08:21,422 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:08:21,534 INFO] number of examples: 10264
[2020-02-09 08:08:31,912 INFO] Validation perplexity: 2.23017
[2020-02-09 08:08:31,912 INFO] Validation accuracy: 86.3763
[2020-02-09 08:08:32,381 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_4500.pt
[2020-02-09 08:08:59,096 INFO] Step 4550/100000; acc:  93.64; ppl:  1.36; xent: 0.31; lr: 0.00056; 9942/10391 tok/s;   2396 sec
[2020-02-09 08:09:23,448 INFO] Step 4600/100000; acc:  95.06; ppl:  1.27; xent: 0.24; lr: 0.00057; 15208/15990 tok/s;   2421 sec
[2020-02-09 08:09:48,118 INFO] Step 4650/100000; acc:  95.30; ppl:  1.25; xent: 0.23; lr: 0.00057; 15129/15846 tok/s;   2445 sec
[2020-02-09 08:10:12,847 INFO] Step 4700/100000; acc:  95.41; ppl:  1.25; xent: 0.22; lr: 0.00058; 15177/15843 tok/s;   2470 sec
[2020-02-09 08:10:37,333 INFO] Step 4750/100000; acc:  95.44; ppl:  1.25; xent: 0.22; lr: 0.00059; 15139/15910 tok/s;   2495 sec
[2020-02-09 08:11:01,817 INFO] Step 4800/100000; acc:  95.20; ppl:  1.26; xent: 0.23; lr: 0.00059; 15093/15882 tok/s;   2519 sec
[2020-02-09 08:11:26,008 INFO] Step 4850/100000; acc:  95.68; ppl:  1.23; xent: 0.21; lr: 0.00060; 15255/16068 tok/s;   2543 sec
[2020-02-09 08:11:50,315 INFO] Step 4900/100000; acc:  95.22; ppl:  1.26; xent: 0.23; lr: 0.00061; 15186/15992 tok/s;   2568 sec
[2020-02-09 08:12:14,770 INFO] Step 4950/100000; acc:  94.68; ppl:  1.30; xent: 0.26; lr: 0.00061; 15171/15938 tok/s;   2592 sec
[2020-02-09 08:12:39,503 INFO] Step 5000/100000; acc:  95.50; ppl:  1.24; xent: 0.22; lr: 0.00062; 15169/15839 tok/s;   2617 sec
[2020-02-09 08:12:39,505 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:12:39,615 INFO] number of examples: 10264
[2020-02-09 08:12:49,962 INFO] Validation perplexity: 1.9532
[2020-02-09 08:12:49,962 INFO] Validation accuracy: 88.5341
[2020-02-09 08:12:50,432 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_5000.pt
[2020-02-09 08:13:04,830 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 08:13:12,247 INFO] number of examples: 368702
[2020-02-09 08:13:28,495 INFO] Step 5050/100000; acc:  95.78; ppl:  1.22; xent: 0.20; lr: 0.00062; 7628/7981 tok/s;   2666 sec
[2020-02-09 08:13:52,993 INFO] Step 5100/100000; acc:  95.56; ppl:  1.23; xent: 0.21; lr: 0.00063; 15149/15908 tok/s;   2690 sec
[2020-02-09 08:14:17,477 INFO] Step 5150/100000; acc:  95.89; ppl:  1.22; xent: 0.20; lr: 0.00064; 15175/15918 tok/s;   2715 sec
[2020-02-09 08:14:42,018 INFO] Step 5200/100000; acc:  95.93; ppl:  1.21; xent: 0.19; lr: 0.00064; 15174/15906 tok/s;   2739 sec
[2020-02-09 08:15:06,627 INFO] Step 5250/100000; acc:  95.97; ppl:  1.21; xent: 0.19; lr: 0.00065; 15130/15863 tok/s;   2764 sec
[2020-02-09 08:15:31,163 INFO] Step 5300/100000; acc:  95.95; ppl:  1.21; xent: 0.19; lr: 0.00065; 15124/15883 tok/s;   2788 sec
[2020-02-09 08:15:55,301 INFO] Step 5350/100000; acc:  95.74; ppl:  1.23; xent: 0.20; lr: 0.00066; 15226/16074 tok/s;   2813 sec
[2020-02-09 08:16:19,627 INFO] Step 5400/100000; acc:  95.93; ppl:  1.21; xent: 0.19; lr: 0.00067; 15207/16001 tok/s;   2837 sec
[2020-02-09 08:16:44,229 INFO] Step 5450/100000; acc:  96.20; ppl:  1.20; xent: 0.18; lr: 0.00067; 15173/15883 tok/s;   2862 sec
[2020-02-09 08:17:08,527 INFO] Step 5500/100000; acc:  95.91; ppl:  1.21; xent: 0.19; lr: 0.00068; 15203/15961 tok/s;   2886 sec
[2020-02-09 08:17:08,529 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:17:08,641 INFO] number of examples: 10264
[2020-02-09 08:17:19,035 INFO] Validation perplexity: 1.90959
[2020-02-09 08:17:19,035 INFO] Validation accuracy: 88.8646
[2020-02-09 08:17:19,502 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_5500.pt
[2020-02-09 08:17:46,130 INFO] Step 5550/100000; acc:  96.02; ppl:  1.21; xent: 0.19; lr: 0.00069; 9962/10411 tok/s;   2923 sec
[2020-02-09 08:18:10,568 INFO] Step 5600/100000; acc:  96.08; ppl:  1.20; xent: 0.19; lr: 0.00069; 15169/15940 tok/s;   2948 sec
[2020-02-09 08:18:35,241 INFO] Step 5650/100000; acc:  96.15; ppl:  1.20; xent: 0.18; lr: 0.00070; 15164/15862 tok/s;   2973 sec
[2020-02-09 08:18:59,860 INFO] Step 5700/100000; acc:  96.04; ppl:  1.20; xent: 0.19; lr: 0.00070; 15209/15895 tok/s;   2997 sec
[2020-02-09 08:19:24,348 INFO] Step 5750/100000; acc:  95.94; ppl:  1.21; xent: 0.19; lr: 0.00071; 15192/15938 tok/s;   3022 sec
[2020-02-09 08:19:48,842 INFO] Step 5800/100000; acc:  95.83; ppl:  1.22; xent: 0.20; lr: 0.00072; 15082/15877 tok/s;   3046 sec
[2020-02-09 08:20:13,040 INFO] Step 5850/100000; acc:  96.20; ppl:  1.20; xent: 0.18; lr: 0.00072; 15235/16049 tok/s;   3070 sec
[2020-02-09 08:20:37,169 INFO] Step 5900/100000; acc:  96.01; ppl:  1.21; xent: 0.19; lr: 0.00073; 15256/16090 tok/s;   3095 sec
[2020-02-09 08:21:01,721 INFO] Step 5950/100000; acc:  96.12; ppl:  1.20; xent: 0.19; lr: 0.00074; 15163/15900 tok/s;   3119 sec
[2020-02-09 08:21:26,267 INFO] Step 6000/100000; acc:  96.33; ppl:  1.19; xent: 0.17; lr: 0.00074; 15235/15938 tok/s;   3144 sec
[2020-02-09 08:21:26,269 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:21:26,381 INFO] number of examples: 10264
[2020-02-09 08:21:36,721 INFO] Validation perplexity: 1.94879
[2020-02-09 08:21:36,721 INFO] Validation accuracy: 89.0684
[2020-02-09 08:21:37,188 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_6000.pt
[2020-02-09 08:21:54,052 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 08:22:00,050 INFO] number of examples: 368702
[2020-02-09 08:22:13,869 INFO] Step 6050/100000; acc:  96.43; ppl:  1.18; xent: 0.17; lr: 0.00075; 7857/8215 tok/s;   3191 sec
[2020-02-09 08:22:38,260 INFO] Step 6100/100000; acc:  96.27; ppl:  1.19; xent: 0.17; lr: 0.00075; 15189/15965 tok/s;   3216 sec
[2020-02-09 08:23:02,893 INFO] Step 6150/100000; acc:  96.37; ppl:  1.18; xent: 0.17; lr: 0.00076; 15166/15868 tok/s;   3240 sec
[2020-02-09 08:23:27,284 INFO] Step 6200/100000; acc:  96.42; ppl:  1.18; xent: 0.17; lr: 0.00077; 15190/15960 tok/s;   3265 sec
[2020-02-09 08:23:51,921 INFO] Step 6250/100000; acc:  96.51; ppl:  1.18; xent: 0.16; lr: 0.00077; 15144/15861 tok/s;   3289 sec
[2020-02-09 08:24:16,484 INFO] Step 6300/100000; acc:  96.46; ppl:  1.18; xent: 0.17; lr: 0.00078; 15100/15863 tok/s;   3314 sec
[2020-02-09 08:24:40,641 INFO] Step 6350/100000; acc:  96.39; ppl:  1.18; xent: 0.17; lr: 0.00078; 15227/16066 tok/s;   3338 sec
[2020-02-09 08:25:04,861 INFO] Step 6400/100000; acc:  96.51; ppl:  1.18; xent: 0.16; lr: 0.00079; 15236/16053 tok/s;   3362 sec
[2020-02-09 08:25:29,495 INFO] Step 6450/100000; acc:  96.73; ppl:  1.17; xent: 0.15; lr: 0.00080; 15175/15873 tok/s;   3387 sec
[2020-02-09 08:25:53,791 INFO] Step 6500/100000; acc:  96.48; ppl:  1.18; xent: 0.16; lr: 0.00080; 15224/15972 tok/s;   3411 sec
[2020-02-09 08:25:53,793 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:25:56,196 INFO] number of examples: 10264
[2020-02-09 08:26:06,563 INFO] Validation perplexity: 1.85805
[2020-02-09 08:26:06,563 INFO] Validation accuracy: 89.6948
[2020-02-09 08:26:07,026 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_6500.pt
[2020-02-09 08:26:33,549 INFO] Step 6550/100000; acc:  96.49; ppl:  1.18; xent: 0.16; lr: 0.00081; 9406/9839 tok/s;   3451 sec
[2020-02-09 08:26:57,987 INFO] Step 6600/100000; acc:  96.66; ppl:  1.17; xent: 0.16; lr: 0.00082; 15229/15975 tok/s;   3475 sec
[2020-02-09 08:27:22,589 INFO] Step 6650/100000; acc:  96.58; ppl:  1.17; xent: 0.16; lr: 0.00082; 15127/15863 tok/s;   3500 sec
[2020-02-09 08:27:47,265 INFO] Step 6700/100000; acc:  96.62; ppl:  1.17; xent: 0.16; lr: 0.00083; 15191/15868 tok/s;   3525 sec
[2020-02-09 08:28:11,745 INFO] Step 6750/100000; acc:  96.70; ppl:  1.17; xent: 0.15; lr: 0.00083; 15179/15934 tok/s;   3549 sec
[2020-02-09 08:28:36,335 INFO] Step 6800/100000; acc:  96.44; ppl:  1.18; xent: 0.17; lr: 0.00084; 15072/15839 tok/s;   3574 sec
[2020-02-09 08:29:00,530 INFO] Step 6850/100000; acc:  96.67; ppl:  1.17; xent: 0.15; lr: 0.00085; 15247/16056 tok/s;   3598 sec
[2020-02-09 08:29:24,643 INFO] Step 6900/100000; acc:  96.73; ppl:  1.17; xent: 0.15; lr: 0.00085; 15293/16113 tok/s;   3622 sec
[2020-02-09 08:29:49,055 INFO] Step 6950/100000; acc:  96.23; ppl:  1.19; xent: 0.18; lr: 0.00086; 15181/15957 tok/s;   3646 sec
[2020-02-09 08:30:13,603 INFO] Step 7000/100000; acc:  96.63; ppl:  1.17; xent: 0.16; lr: 0.00086; 15250/15946 tok/s;   3671 sec
[2020-02-09 08:30:13,605 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:30:13,704 INFO] number of examples: 10264
[2020-02-09 08:30:24,020 INFO] Validation perplexity: 2.01566
[2020-02-09 08:30:24,020 INFO] Validation accuracy: 88.8228
[2020-02-09 08:30:24,485 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_7000.pt
[2020-02-09 08:30:43,788 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 08:30:56,733 INFO] number of examples: 368702
[2020-02-09 08:31:08,353 INFO] Step 7050/100000; acc:  96.86; ppl:  1.16; xent: 0.15; lr: 0.00087; 6833/7143 tok/s;   3726 sec
[2020-02-09 08:31:32,672 INFO] Step 7100/100000; acc:  96.94; ppl:  1.15; xent: 0.14; lr: 0.00088; 15236/16015 tok/s;   3750 sec
[2020-02-09 08:31:57,371 INFO] Step 7150/100000; acc:  96.87; ppl:  1.16; xent: 0.15; lr: 0.00088; 15155/15840 tok/s;   3775 sec
[2020-02-09 08:32:21,724 INFO] Step 7200/100000; acc:  96.90; ppl:  1.16; xent: 0.14; lr: 0.00089; 15179/15967 tok/s;   3799 sec
[2020-02-09 08:32:46,274 INFO] Step 7250/100000; acc:  97.08; ppl:  1.15; xent: 0.14; lr: 0.00090; 15184/15909 tok/s;   3824 sec
[2020-02-09 08:33:10,793 INFO] Step 7300/100000; acc:  97.06; ppl:  1.15; xent: 0.14; lr: 0.00090; 15124/15890 tok/s;   3848 sec
[2020-02-09 08:33:35,106 INFO] Step 7350/100000; acc:  96.95; ppl:  1.15; xent: 0.14; lr: 0.00091; 15170/15984 tok/s;   3872 sec
[2020-02-09 08:33:59,312 INFO] Step 7400/100000; acc:  97.11; ppl:  1.14; xent: 0.13; lr: 0.00091; 15224/16052 tok/s;   3897 sec
[2020-02-09 08:34:23,927 INFO] Step 7450/100000; acc:  97.18; ppl:  1.14; xent: 0.13; lr: 0.00092; 15182/15886 tok/s;   3921 sec
[2020-02-09 08:34:48,289 INFO] Step 7500/100000; acc:  97.06; ppl:  1.15; xent: 0.14; lr: 0.00093; 15184/15926 tok/s;   3946 sec
[2020-02-09 08:34:48,291 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:34:48,401 INFO] number of examples: 10264
[2020-02-09 08:34:58,792 INFO] Validation perplexity: 1.88012
[2020-02-09 08:34:58,792 INFO] Validation accuracy: 89.5544
[2020-02-09 08:34:59,262 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_7500.pt
[2020-02-09 08:35:25,865 INFO] Step 7550/100000; acc:  97.04; ppl:  1.15; xent: 0.14; lr: 0.00093; 9941/10405 tok/s;   3983 sec
[2020-02-09 08:35:50,363 INFO] Step 7600/100000; acc:  97.14; ppl:  1.14; xent: 0.13; lr: 0.00094; 15191/15935 tok/s;   4008 sec
[2020-02-09 08:36:15,002 INFO] Step 7650/100000; acc:  97.13; ppl:  1.14; xent: 0.13; lr: 0.00095; 15115/15843 tok/s;   4032 sec
[2020-02-09 08:36:39,655 INFO] Step 7700/100000; acc:  97.21; ppl:  1.14; xent: 0.13; lr: 0.00095; 15203/15884 tok/s;   4057 sec
[2020-02-09 08:37:04,017 INFO] Step 7750/100000; acc:  97.26; ppl:  1.14; xent: 0.13; lr: 0.00096; 15221/15995 tok/s;   4081 sec
[2020-02-09 08:37:28,673 INFO] Step 7800/100000; acc:  96.98; ppl:  1.15; xent: 0.14; lr: 0.00096; 15054/15808 tok/s;   4106 sec
[2020-02-09 08:37:52,880 INFO] Step 7850/100000; acc:  97.24; ppl:  1.14; xent: 0.13; lr: 0.00097; 15240/16048 tok/s;   4130 sec
[2020-02-09 08:38:17,030 INFO] Step 7900/100000; acc:  97.25; ppl:  1.14; xent: 0.13; lr: 0.00098; 15261/16084 tok/s;   4154 sec
[2020-02-09 08:38:41,569 INFO] Step 7950/100000; acc:  96.58; ppl:  1.18; xent: 0.16; lr: 0.00098; 15151/15899 tok/s;   4179 sec
[2020-02-09 08:39:06,128 INFO] Step 8000/100000; acc:  96.74; ppl:  1.17; xent: 0.15; lr: 0.00099; 15213/15923 tok/s;   4203 sec
[2020-02-09 08:39:06,130 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:39:06,243 INFO] number of examples: 10264
[2020-02-09 08:39:16,586 INFO] Validation perplexity: 1.89065
[2020-02-09 08:39:16,586 INFO] Validation accuracy: 89.6512
[2020-02-09 08:39:17,048 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_8000.pt
[2020-02-09 08:39:38,787 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 08:39:46,085 INFO] number of examples: 368702
[2020-02-09 08:39:55,007 INFO] Step 8050/100000; acc:  97.24; ppl:  1.14; xent: 0.13; lr: 0.00099; 7653/8001 tok/s;   4252 sec
[2020-02-09 08:40:19,382 INFO] Step 8100/100000; acc:  97.27; ppl:  1.14; xent: 0.13; lr: 0.00098; 15222/15989 tok/s;   4277 sec
[2020-02-09 08:40:44,004 INFO] Step 8150/100000; acc:  97.29; ppl:  1.14; xent: 0.13; lr: 0.00098; 15161/15870 tok/s;   4301 sec
[2020-02-09 08:41:08,319 INFO] Step 8200/100000; acc:  97.43; ppl:  1.13; xent: 0.12; lr: 0.00098; 15197/15988 tok/s;   4326 sec
[2020-02-09 08:41:33,016 INFO] Step 8250/100000; acc:  97.52; ppl:  1.13; xent: 0.12; lr: 0.00097; 15160/15851 tok/s;   4350 sec
[2020-02-09 08:41:57,519 INFO] Step 8300/100000; acc:  97.51; ppl:  1.13; xent: 0.12; lr: 0.00097; 15123/15892 tok/s;   4375 sec
[2020-02-09 08:42:21,910 INFO] Step 8350/100000; acc:  97.45; ppl:  1.13; xent: 0.12; lr: 0.00097; 15139/15942 tok/s;   4399 sec
[2020-02-09 08:42:46,027 INFO] Step 8400/100000; acc:  97.59; ppl:  1.12; xent: 0.11; lr: 0.00096; 15268/16105 tok/s;   4423 sec
[2020-02-09 08:43:10,509 INFO] Step 8450/100000; acc:  97.71; ppl:  1.12; xent: 0.11; lr: 0.00096; 15197/15939 tok/s;   4448 sec
[2020-02-09 08:43:34,918 INFO] Step 8500/100000; acc:  97.52; ppl:  1.12; xent: 0.12; lr: 0.00096; 15185/15910 tok/s;   4472 sec
[2020-02-09 08:43:34,920 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:43:35,032 INFO] number of examples: 10264
[2020-02-09 08:43:45,407 INFO] Validation perplexity: 2.0087
[2020-02-09 08:43:45,408 INFO] Validation accuracy: 89.2533
[2020-02-09 08:43:45,877 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_8500.pt
[2020-02-09 08:44:12,414 INFO] Step 8550/100000; acc:  97.33; ppl:  1.14; xent: 0.13; lr: 0.00096; 9963/10428 tok/s;   4510 sec
[2020-02-09 08:44:36,962 INFO] Step 8600/100000; acc:  97.57; ppl:  1.12; xent: 0.12; lr: 0.00095; 15198/15921 tok/s;   4534 sec
[2020-02-09 08:45:01,458 INFO] Step 8650/100000; acc:  97.64; ppl:  1.12; xent: 0.11; lr: 0.00095; 15155/15911 tok/s;   4559 sec
[2020-02-09 08:45:26,021 INFO] Step 8700/100000; acc:  97.70; ppl:  1.12; xent: 0.11; lr: 0.00095; 15226/15926 tok/s;   4583 sec
[2020-02-09 08:45:50,558 INFO] Step 8750/100000; acc:  97.74; ppl:  1.11; xent: 0.11; lr: 0.00094; 15177/15913 tok/s;   4608 sec
[2020-02-09 08:46:15,069 INFO] Step 8800/100000; acc:  97.64; ppl:  1.12; xent: 0.11; lr: 0.00094; 15144/15902 tok/s;   4632 sec
[2020-02-09 08:46:39,302 INFO] Step 8850/100000; acc:  97.78; ppl:  1.11; xent: 0.11; lr: 0.00094; 15203/16021 tok/s;   4657 sec
[2020-02-09 08:47:03,433 INFO] Step 8900/100000; acc:  97.78; ppl:  1.11; xent: 0.11; lr: 0.00094; 15274/16097 tok/s;   4681 sec
[2020-02-09 08:47:28,007 INFO] Step 8950/100000; acc:  97.71; ppl:  1.12; xent: 0.11; lr: 0.00093; 15153/15887 tok/s;   4705 sec
[2020-02-09 08:47:52,428 INFO] Step 9000/100000; acc:  97.82; ppl:  1.11; xent: 0.10; lr: 0.00093; 15253/15990 tok/s;   4730 sec
[2020-02-09 08:47:52,429 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:47:52,541 INFO] number of examples: 10264
[2020-02-09 08:48:02,882 INFO] Validation perplexity: 1.80626
[2020-02-09 08:48:02,882 INFO] Validation accuracy: 90.3252
[2020-02-09 08:48:03,350 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_9000.pt
[2020-02-09 08:48:27,539 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 08:48:34,970 INFO] number of examples: 368702
[2020-02-09 08:48:41,424 INFO] Step 9050/100000; acc:  97.96; ppl:  1.10; xent: 0.10; lr: 0.00093; 7640/7984 tok/s;   4779 sec
[2020-02-09 08:49:05,837 INFO] Step 9100/100000; acc:  97.94; ppl:  1.10; xent: 0.10; lr: 0.00093; 15201/15964 tok/s;   4803 sec
[2020-02-09 08:49:30,480 INFO] Step 9150/100000; acc:  97.88; ppl:  1.11; xent: 0.10; lr: 0.00092; 15169/15870 tok/s;   4828 sec
[2020-02-09 08:49:54,865 INFO] Step 9200/100000; acc:  97.94; ppl:  1.10; xent: 0.10; lr: 0.00092; 15180/15953 tok/s;   4852 sec
[2020-02-09 08:50:19,483 INFO] Step 9250/100000; acc:  97.93; ppl:  1.10; xent: 0.10; lr: 0.00092; 15187/15892 tok/s;   4877 sec
[2020-02-09 08:50:44,027 INFO] Step 9300/100000; acc:  97.88; ppl:  1.11; xent: 0.10; lr: 0.00092; 15093/15862 tok/s;   4901 sec
[2020-02-09 08:51:08,322 INFO] Step 9350/100000; acc:  97.89; ppl:  1.10; xent: 0.10; lr: 0.00091; 15197/16004 tok/s;   4926 sec
[2020-02-09 08:51:32,435 INFO] Step 9400/100000; acc:  97.92; ppl:  1.10; xent: 0.10; lr: 0.00091; 15262/16103 tok/s;   4950 sec
[2020-02-09 08:51:56,921 INFO] Step 9450/100000; acc:  98.04; ppl:  1.10; xent: 0.09; lr: 0.00091; 15199/15939 tok/s;   4974 sec
[2020-02-09 08:52:21,237 INFO] Step 9500/100000; acc:  98.05; ppl:  1.10; xent: 0.09; lr: 0.00091; 15204/15951 tok/s;   4999 sec
[2020-02-09 08:52:21,239 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:52:21,339 INFO] number of examples: 10264
[2020-02-09 08:52:31,692 INFO] Validation perplexity: 1.96424
[2020-02-09 08:52:31,692 INFO] Validation accuracy: 89.7234
[2020-02-09 08:52:32,152 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_9500.pt
[2020-02-09 08:52:58,743 INFO] Step 9550/100000; acc:  98.00; ppl:  1.10; xent: 0.10; lr: 0.00090; 9976/10432 tok/s;   5036 sec
[2020-02-09 08:53:23,266 INFO] Step 9600/100000; acc:  98.04; ppl:  1.10; xent: 0.09; lr: 0.00090; 15204/15933 tok/s;   5061 sec
[2020-02-09 08:53:47,806 INFO] Step 9650/100000; acc:  98.02; ppl:  1.10; xent: 0.09; lr: 0.00090; 15152/15895 tok/s;   5085 sec
[2020-02-09 08:54:12,348 INFO] Step 9700/100000; acc:  98.00; ppl:  1.10; xent: 0.10; lr: 0.00090; 15218/15930 tok/s;   5110 sec
[2020-02-09 08:54:37,031 INFO] Step 9750/100000; acc:  98.06; ppl:  1.10; xent: 0.09; lr: 0.00090; 15188/15868 tok/s;   5134 sec
[2020-02-09 08:55:01,491 INFO] Step 9800/100000; acc:  97.95; ppl:  1.10; xent: 0.10; lr: 0.00089; 15106/15902 tok/s;   5159 sec
[2020-02-09 08:55:25,805 INFO] Step 9850/100000; acc:  98.08; ppl:  1.10; xent: 0.09; lr: 0.00089; 15186/15984 tok/s;   5183 sec
[2020-02-09 08:55:49,926 INFO] Step 9900/100000; acc:  98.11; ppl:  1.10; xent: 0.09; lr: 0.00089; 15262/16094 tok/s;   5207 sec
[2020-02-09 08:56:14,526 INFO] Step 9950/100000; acc:  97.95; ppl:  1.10; xent: 0.10; lr: 0.00089; 15096/15851 tok/s;   5232 sec
[2020-02-09 08:56:38,985 INFO] Step 10000/100000; acc:  98.17; ppl:  1.09; xent: 0.09; lr: 0.00088; 15227/15965 tok/s;   5256 sec
[2020-02-09 08:56:38,987 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 08:56:39,100 INFO] number of examples: 10264
[2020-02-09 08:56:49,461 INFO] Validation perplexity: 1.85077
[2020-02-09 08:56:49,461 INFO] Validation accuracy: 90.2186
[2020-02-09 08:56:49,930 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_10000.pt
[2020-02-09 08:57:16,647 INFO] Step 10050/100000; acc:  98.17; ppl:  1.09; xent: 0.09; lr: 0.00088; 9939/10387 tok/s;   5294 sec
[2020-02-09 08:57:16,685 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 08:57:24,121 INFO] number of examples: 368702
[2020-02-09 08:57:52,588 INFO] Step 10100/100000; acc:  98.14; ppl:  1.09; xent: 0.09; lr: 0.00088; 10327/10845 tok/s;   5330 sec
[2020-02-09 08:58:17,219 INFO] Step 10150/100000; acc:  98.13; ppl:  1.09; xent: 0.09; lr: 0.00088; 15175/15876 tok/s;   5355 sec
[2020-02-09 08:58:41,615 INFO] Step 10200/100000; acc:  98.21; ppl:  1.09; xent: 0.09; lr: 0.00088; 15195/15957 tok/s;   5379 sec
[2020-02-09 08:59:06,216 INFO] Step 10250/100000; acc:  98.22; ppl:  1.09; xent: 0.09; lr: 0.00087; 15164/15886 tok/s;   5404 sec
[2020-02-09 08:59:30,839 INFO] Step 10300/100000; acc:  98.19; ppl:  1.09; xent: 0.09; lr: 0.00087; 15102/15842 tok/s;   5428 sec
[2020-02-09 08:59:55,212 INFO] Step 10350/100000; acc:  98.19; ppl:  1.09; xent: 0.09; lr: 0.00087; 15185/15969 tok/s;   5453 sec
[2020-02-09 09:00:19,284 INFO] Step 10400/100000; acc:  98.25; ppl:  1.09; xent: 0.09; lr: 0.00087; 15242/16108 tok/s;   5477 sec
[2020-02-09 09:00:43,812 INFO] Step 10450/100000; acc:  98.28; ppl:  1.09; xent: 0.08; lr: 0.00086; 15184/15918 tok/s;   5501 sec
[2020-02-09 09:01:08,130 INFO] Step 10500/100000; acc:  98.26; ppl:  1.09; xent: 0.09; lr: 0.00086; 15192/15944 tok/s;   5525 sec
[2020-02-09 09:01:08,132 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:01:08,244 INFO] number of examples: 10264
[2020-02-09 09:01:18,614 INFO] Validation perplexity: 1.87704
[2020-02-09 09:01:18,614 INFO] Validation accuracy: 90.212
[2020-02-09 09:01:19,080 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_10500.pt
[2020-02-09 09:01:45,535 INFO] Step 10550/100000; acc:  98.24; ppl:  1.09; xent: 0.09; lr: 0.00086; 9973/10446 tok/s;   5563 sec
[2020-02-09 09:02:10,073 INFO] Step 10600/100000; acc:  98.28; ppl:  1.09; xent: 0.08; lr: 0.00086; 15190/15918 tok/s;   5587 sec
[2020-02-09 09:02:34,668 INFO] Step 10650/100000; acc:  98.29; ppl:  1.09; xent: 0.08; lr: 0.00086; 15174/15888 tok/s;   5612 sec
[2020-02-09 09:02:59,103 INFO] Step 10700/100000; acc:  98.29; ppl:  1.09; xent: 0.08; lr: 0.00085; 15240/15980 tok/s;   5636 sec
[2020-02-09 09:03:23,705 INFO] Step 10750/100000; acc:  98.37; ppl:  1.08; xent: 0.08; lr: 0.00085; 15215/15908 tok/s;   5661 sec
[2020-02-09 09:03:48,292 INFO] Step 10800/100000; acc:  98.19; ppl:  1.09; xent: 0.09; lr: 0.00085; 15116/15862 tok/s;   5686 sec
[2020-02-09 09:04:12,587 INFO] Step 10850/100000; acc:  98.22; ppl:  1.09; xent: 0.09; lr: 0.00085; 15171/15983 tok/s;   5710 sec
[2020-02-09 09:04:36,660 INFO] Step 10900/100000; acc:  98.32; ppl:  1.09; xent: 0.08; lr: 0.00085; 15277/16119 tok/s;   5734 sec
[2020-02-09 09:05:01,157 INFO] Step 10950/100000; acc:  98.23; ppl:  1.09; xent: 0.09; lr: 0.00084; 15152/15914 tok/s;   5758 sec
[2020-02-09 09:05:25,517 INFO] Step 11000/100000; acc:  98.38; ppl:  1.08; xent: 0.08; lr: 0.00084; 15282/16027 tok/s;   5783 sec
[2020-02-09 09:05:25,518 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:05:25,629 INFO] number of examples: 10264
[2020-02-09 09:05:35,932 INFO] Validation perplexity: 1.79019
[2020-02-09 09:05:35,932 INFO] Validation accuracy: 90.7042
[2020-02-09 09:05:36,398 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_11000.pt
[2020-02-09 09:06:02,989 INFO] Step 11050/100000; acc:  98.44; ppl:  1.08; xent: 0.08; lr: 0.00084; 9981/10439 tok/s;   5820 sec
[2020-02-09 09:06:05,534 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 09:06:13,428 INFO] number of examples: 368702
[2020-02-09 09:06:39,622 INFO] Step 11100/100000; acc:  98.45; ppl:  1.08; xent: 0.08; lr: 0.00084; 10173/10656 tok/s;   5857 sec
[2020-02-09 09:07:04,048 INFO] Step 11150/100000; acc:  98.34; ppl:  1.08; xent: 0.08; lr: 0.00084; 15221/15970 tok/s;   5881 sec
[2020-02-09 09:07:28,545 INFO] Step 11200/100000; acc:  98.43; ppl:  1.08; xent: 0.08; lr: 0.00084; 15188/15919 tok/s;   5906 sec
[2020-02-09 09:07:53,020 INFO] Step 11250/100000; acc:  98.44; ppl:  1.08; xent: 0.08; lr: 0.00083; 15215/15954 tok/s;   5930 sec
[2020-02-09 09:08:17,603 INFO] Step 11300/100000; acc:  98.44; ppl:  1.08; xent: 0.08; lr: 0.00083; 15172/15890 tok/s;   5955 sec
[2020-02-09 09:08:41,969 INFO] Step 11350/100000; acc:  98.46; ppl:  1.08; xent: 0.08; lr: 0.00083; 15175/15969 tok/s;   5979 sec
[2020-02-09 09:09:06,036 INFO] Step 11400/100000; acc:  98.43; ppl:  1.08; xent: 0.08; lr: 0.00083; 15255/16113 tok/s;   6003 sec
[2020-02-09 09:09:30,402 INFO] Step 11450/100000; acc:  98.53; ppl:  1.08; xent: 0.07; lr: 0.00083; 15223/15993 tok/s;   6028 sec
[2020-02-09 09:09:54,839 INFO] Step 11500/100000; acc:  98.49; ppl:  1.08; xent: 0.08; lr: 0.00082; 15164/15890 tok/s;   6052 sec
[2020-02-09 09:09:54,841 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:09:54,953 INFO] number of examples: 10264
[2020-02-09 09:10:05,299 INFO] Validation perplexity: 1.95925
[2020-02-09 09:10:05,299 INFO] Validation accuracy: 90.0804
[2020-02-09 09:10:05,770 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_11500.pt
[2020-02-09 09:10:32,106 INFO] Step 11550/100000; acc:  98.48; ppl:  1.08; xent: 0.08; lr: 0.00082; 9981/10471 tok/s;   6089 sec
[2020-02-09 09:10:56,740 INFO] Step 11600/100000; acc:  98.44; ppl:  1.08; xent: 0.08; lr: 0.00082; 15185/15883 tok/s;   6114 sec
[2020-02-09 09:11:21,232 INFO] Step 11650/100000; acc:  98.46; ppl:  1.08; xent: 0.08; lr: 0.00082; 15196/15934 tok/s;   6139 sec
[2020-02-09 09:11:45,732 INFO] Step 11700/100000; acc:  98.46; ppl:  1.08; xent: 0.08; lr: 0.00082; 15228/15953 tok/s;   6163 sec
[2020-02-09 09:12:10,363 INFO] Step 11750/100000; acc:  98.56; ppl:  1.08; xent: 0.07; lr: 0.00082; 15210/15894 tok/s;   6188 sec
[2020-02-09 09:12:34,935 INFO] Step 11800/100000; acc:  98.46; ppl:  1.08; xent: 0.08; lr: 0.00081; 15115/15866 tok/s;   6212 sec
[2020-02-09 09:12:59,240 INFO] Step 11850/100000; acc:  98.47; ppl:  1.08; xent: 0.08; lr: 0.00081; 15169/15979 tok/s;   6237 sec
[2020-02-09 09:13:23,297 INFO] Step 11900/100000; acc:  98.55; ppl:  1.08; xent: 0.07; lr: 0.00081; 15288/16137 tok/s;   6261 sec
[2020-02-09 09:13:47,741 INFO] Step 11950/100000; acc:  98.48; ppl:  1.08; xent: 0.08; lr: 0.00081; 15152/15926 tok/s;   6285 sec
[2020-02-09 09:14:12,144 INFO] Step 12000/100000; acc:  98.55; ppl:  1.08; xent: 0.07; lr: 0.00081; 15262/16003 tok/s;   6309 sec
[2020-02-09 09:14:12,146 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:14:12,257 INFO] number of examples: 10264
[2020-02-09 09:14:22,592 INFO] Validation perplexity: 1.8724
[2020-02-09 09:14:22,592 INFO] Validation accuracy: 90.3793
[2020-02-09 09:14:23,059 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_12000.pt
[2020-02-09 09:14:49,646 INFO] Step 12050/100000; acc:  98.64; ppl:  1.07; xent: 0.07; lr: 0.00081; 9951/10419 tok/s;   6347 sec
[2020-02-09 09:14:54,705 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 09:15:00,778 INFO] number of examples: 368702
[2020-02-09 09:15:24,338 INFO] Step 12100/100000; acc:  98.62; ppl:  1.07; xent: 0.07; lr: 0.00080; 10777/11270 tok/s;   6382 sec
[2020-02-09 09:15:48,719 INFO] Step 12150/100000; acc:  98.55; ppl:  1.08; xent: 0.07; lr: 0.00080; 15208/15979 tok/s;   6406 sec
[2020-02-09 09:16:13,327 INFO] Step 12200/100000; acc:  98.59; ppl:  1.07; xent: 0.07; lr: 0.00080; 15161/15871 tok/s;   6431 sec
[2020-02-09 09:16:37,780 INFO] Step 12250/100000; acc:  98.66; ppl:  1.07; xent: 0.07; lr: 0.00080; 15210/15955 tok/s;   6455 sec
[2020-02-09 09:17:02,349 INFO] Step 12300/100000; acc:  98.62; ppl:  1.07; xent: 0.07; lr: 0.00080; 15167/15893 tok/s;   6480 sec
[2020-02-09 09:17:26,683 INFO] Step 12350/100000; acc:  98.63; ppl:  1.07; xent: 0.07; lr: 0.00080; 15185/15984 tok/s;   6504 sec
[2020-02-09 09:17:50,828 INFO] Step 12400/100000; acc:  98.62; ppl:  1.07; xent: 0.07; lr: 0.00079; 15271/16094 tok/s;   6528 sec
[2020-02-09 09:18:15,140 INFO] Step 12450/100000; acc:  98.67; ppl:  1.07; xent: 0.07; lr: 0.00079; 15207/16004 tok/s;   6552 sec
[2020-02-09 09:18:39,597 INFO] Step 12500/100000; acc:  98.70; ppl:  1.07; xent: 0.07; lr: 0.00079; 15174/15887 tok/s;   6577 sec
[2020-02-09 09:18:39,599 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:18:39,710 INFO] number of examples: 10264
[2020-02-09 09:18:50,057 INFO] Validation perplexity: 1.88698
[2020-02-09 09:18:50,058 INFO] Validation accuracy: 90.4084
[2020-02-09 09:18:50,528 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_12500.pt
[2020-02-09 09:19:16,878 INFO] Step 12550/100000; acc:  98.65; ppl:  1.07; xent: 0.07; lr: 0.00079; 9970/10464 tok/s;   6614 sec
[2020-02-09 09:19:41,589 INFO] Step 12600/100000; acc:  98.66; ppl:  1.07; xent: 0.07; lr: 0.00079; 15167/15847 tok/s;   6639 sec
[2020-02-09 09:20:06,000 INFO] Step 12650/100000; acc:  98.69; ppl:  1.07; xent: 0.07; lr: 0.00079; 15222/15975 tok/s;   6663 sec
[2020-02-09 09:20:30,498 INFO] Step 12700/100000; acc:  98.71; ppl:  1.07; xent: 0.07; lr: 0.00078; 15230/15955 tok/s;   6688 sec
[2020-02-09 09:20:55,137 INFO] Step 12750/100000; acc:  98.74; ppl:  1.07; xent: 0.07; lr: 0.00078; 15200/15885 tok/s;   6712 sec
[2020-02-09 09:21:19,622 INFO] Step 12800/100000; acc:  98.64; ppl:  1.07; xent: 0.07; lr: 0.00078; 15156/15919 tok/s;   6737 sec
[2020-02-09 09:21:44,010 INFO] Step 12850/100000; acc:  98.66; ppl:  1.07; xent: 0.07; lr: 0.00078; 15117/15921 tok/s;   6761 sec
[2020-02-09 09:22:08,058 INFO] Step 12900/100000; acc:  98.70; ppl:  1.07; xent: 0.07; lr: 0.00078; 15285/16139 tok/s;   6785 sec
[2020-02-09 09:22:32,454 INFO] Step 12950/100000; acc:  98.65; ppl:  1.07; xent: 0.07; lr: 0.00078; 15181/15957 tok/s;   6810 sec
[2020-02-09 09:22:56,808 INFO] Step 13000/100000; acc:  98.73; ppl:  1.07; xent: 0.07; lr: 0.00078; 15286/16030 tok/s;   6834 sec
[2020-02-09 09:22:56,810 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:22:59,204 INFO] number of examples: 10264
[2020-02-09 09:23:09,483 INFO] Validation perplexity: 1.92253
[2020-02-09 09:23:09,483 INFO] Validation accuracy: 90.1742
[2020-02-09 09:23:09,944 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_13000.pt
[2020-02-09 09:23:36,589 INFO] Step 13050/100000; acc:  98.82; ppl:  1.07; xent: 0.06; lr: 0.00077; 9422/9843 tok/s;   6874 sec
[2020-02-09 09:23:43,989 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 09:23:57,420 INFO] number of examples: 368702
[2020-02-09 09:24:18,688 INFO] Step 13100/100000; acc:  98.76; ppl:  1.07; xent: 0.06; lr: 0.00077; 8875/9286 tok/s;   6916 sec
[2020-02-09 09:24:43,060 INFO] Step 13150/100000; acc:  98.72; ppl:  1.07; xent: 0.07; lr: 0.00077; 15210/15980 tok/s;   6940 sec
[2020-02-09 09:25:07,593 INFO] Step 13200/100000; acc:  98.75; ppl:  1.07; xent: 0.07; lr: 0.00077; 15182/15908 tok/s;   6965 sec
[2020-02-09 09:25:31,960 INFO] Step 13250/100000; acc:  98.82; ppl:  1.07; xent: 0.06; lr: 0.00077; 15247/16003 tok/s;   6989 sec
[2020-02-09 09:25:56,574 INFO] Step 13300/100000; acc:  98.83; ppl:  1.07; xent: 0.06; lr: 0.00077; 15196/15894 tok/s;   7014 sec
[2020-02-09 09:26:20,888 INFO] Step 13350/100000; acc:  98.72; ppl:  1.07; xent: 0.07; lr: 0.00076; 15169/15981 tok/s;   7038 sec
[2020-02-09 09:26:45,006 INFO] Step 13400/100000; acc:  98.78; ppl:  1.07; xent: 0.06; lr: 0.00076; 15277/16106 tok/s;   7062 sec
[2020-02-09 09:27:09,305 INFO] Step 13450/100000; acc:  98.83; ppl:  1.06; xent: 0.06; lr: 0.00076; 15203/16007 tok/s;   7087 sec
[2020-02-09 09:27:33,834 INFO] Step 13500/100000; acc:  98.84; ppl:  1.06; xent: 0.06; lr: 0.00076; 15209/15926 tok/s;   7111 sec
[2020-02-09 09:27:33,835 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:27:33,946 INFO] number of examples: 10264
[2020-02-09 09:27:44,316 INFO] Validation perplexity: 1.89249
[2020-02-09 09:27:44,316 INFO] Validation accuracy: 90.4317
[2020-02-09 09:27:44,778 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_13500.pt
[2020-02-09 09:28:11,101 INFO] Step 13550/100000; acc:  98.81; ppl:  1.07; xent: 0.06; lr: 0.00076; 9926/10414 tok/s;   7148 sec
[2020-02-09 09:28:35,798 INFO] Step 13600/100000; acc:  98.87; ppl:  1.06; xent: 0.06; lr: 0.00076; 15181/15858 tok/s;   7173 sec
[2020-02-09 09:29:00,127 INFO] Step 13650/100000; acc:  98.81; ppl:  1.06; xent: 0.06; lr: 0.00076; 15239/16012 tok/s;   7197 sec
[2020-02-09 09:29:24,741 INFO] Step 13700/100000; acc:  98.86; ppl:  1.06; xent: 0.06; lr: 0.00076; 15187/15894 tok/s;   7222 sec
[2020-02-09 09:29:49,292 INFO] Step 13750/100000; acc:  98.90; ppl:  1.06; xent: 0.06; lr: 0.00075; 15243/15938 tok/s;   7247 sec
[2020-02-09 09:30:13,701 INFO] Step 13800/100000; acc:  98.86; ppl:  1.06; xent: 0.06; lr: 0.00075; 15199/15965 tok/s;   7271 sec
[2020-02-09 09:30:38,064 INFO] Step 13850/100000; acc:  98.80; ppl:  1.07; xent: 0.06; lr: 0.00075; 15129/15938 tok/s;   7295 sec
[2020-02-09 09:31:02,207 INFO] Step 13900/100000; acc:  98.91; ppl:  1.06; xent: 0.06; lr: 0.00075; 15312/16117 tok/s;   7320 sec
[2020-02-09 09:31:26,469 INFO] Step 13950/100000; acc:  98.81; ppl:  1.07; xent: 0.06; lr: 0.00075; 15197/16012 tok/s;   7344 sec
[2020-02-09 09:31:50,826 INFO] Step 14000/100000; acc:  98.83; ppl:  1.06; xent: 0.06; lr: 0.00075; 15281/16028 tok/s;   7368 sec
[2020-02-09 09:31:50,827 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:31:50,939 INFO] number of examples: 10264
[2020-02-09 09:32:01,242 INFO] Validation perplexity: 1.84526
[2020-02-09 09:32:01,243 INFO] Validation accuracy: 90.5378
[2020-02-09 09:32:01,711 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_14000.pt
[2020-02-09 09:32:28,371 INFO] Step 14050/100000; acc:  98.94; ppl:  1.06; xent: 0.06; lr: 0.00075; 9979/10426 tok/s;   7406 sec
[2020-02-09 09:32:38,285 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 09:32:45,797 INFO] number of examples: 368702
[2020-02-09 09:33:04,448 INFO] Step 14100/100000; acc:  98.92; ppl:  1.06; xent: 0.06; lr: 0.00074; 10350/10834 tok/s;   7442 sec
[2020-02-09 09:33:28,847 INFO] Step 14150/100000; acc:  98.88; ppl:  1.06; xent: 0.06; lr: 0.00074; 15199/15964 tok/s;   7466 sec
[2020-02-09 09:33:53,382 INFO] Step 14200/100000; acc:  98.91; ppl:  1.06; xent: 0.06; lr: 0.00074; 15197/15914 tok/s;   7491 sec
[2020-02-09 09:34:17,795 INFO] Step 14250/100000; acc:  98.93; ppl:  1.06; xent: 0.06; lr: 0.00074; 15220/15972 tok/s;   7515 sec
[2020-02-09 09:34:42,388 INFO] Step 14300/100000; acc:  98.96; ppl:  1.06; xent: 0.06; lr: 0.00074; 15168/15887 tok/s;   7540 sec
[2020-02-09 09:35:06,846 INFO] Step 14350/100000; acc:  98.94; ppl:  1.06; xent: 0.06; lr: 0.00074; 15130/15912 tok/s;   7564 sec
[2020-02-09 09:35:31,033 INFO] Step 14400/100000; acc:  98.95; ppl:  1.06; xent: 0.06; lr: 0.00074; 15216/16052 tok/s;   7588 sec
[2020-02-09 09:35:55,413 INFO] Step 14450/100000; acc:  98.96; ppl:  1.06; xent: 0.06; lr: 0.00074; 15177/15965 tok/s;   7613 sec
[2020-02-09 09:36:19,913 INFO] Step 14500/100000; acc:  99.00; ppl:  1.06; xent: 0.06; lr: 0.00073; 15204/15934 tok/s;   7637 sec
[2020-02-09 09:36:19,915 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:36:20,027 INFO] number of examples: 10264
[2020-02-09 09:36:30,375 INFO] Validation perplexity: 1.91447
[2020-02-09 09:36:30,375 INFO] Validation accuracy: 90.3877
[2020-02-09 09:36:30,845 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_14500.pt
[2020-02-09 09:36:57,221 INFO] Step 14550/100000; acc:  98.96; ppl:  1.06; xent: 0.06; lr: 0.00073; 9936/10412 tok/s;   7675 sec
[2020-02-09 09:37:21,937 INFO] Step 14600/100000; acc:  98.94; ppl:  1.06; xent: 0.06; lr: 0.00073; 15155/15839 tok/s;   7699 sec
[2020-02-09 09:37:46,202 INFO] Step 14650/100000; acc:  99.02; ppl:  1.06; xent: 0.06; lr: 0.00073; 15262/16047 tok/s;   7724 sec
[2020-02-09 09:38:10,809 INFO] Step 14700/100000; acc:  98.95; ppl:  1.06; xent: 0.06; lr: 0.00073; 15169/15887 tok/s;   7748 sec
[2020-02-09 09:38:35,474 INFO] Step 14750/100000; acc:  99.00; ppl:  1.06; xent: 0.06; lr: 0.00073; 15216/15884 tok/s;   7773 sec
[2020-02-09 09:38:59,933 INFO] Step 14800/100000; acc:  98.99; ppl:  1.06; xent: 0.06; lr: 0.00073; 15156/15928 tok/s;   7797 sec
[2020-02-09 09:39:24,372 INFO] Step 14850/100000; acc:  98.89; ppl:  1.06; xent: 0.06; lr: 0.00073; 15120/15910 tok/s;   7822 sec
[2020-02-09 09:39:48,502 INFO] Step 14900/100000; acc:  98.98; ppl:  1.06; xent: 0.06; lr: 0.00072; 15294/16109 tok/s;   7846 sec
[2020-02-09 09:40:12,731 INFO] Step 14950/100000; acc:  98.96; ppl:  1.06; xent: 0.06; lr: 0.00072; 15235/16043 tok/s;   7870 sec
[2020-02-09 09:40:37,099 INFO] Step 15000/100000; acc:  98.89; ppl:  1.06; xent: 0.06; lr: 0.00072; 15225/15995 tok/s;   7894 sec
[2020-02-09 09:40:37,101 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:40:37,212 INFO] number of examples: 10264
[2020-02-09 09:40:47,519 INFO] Validation perplexity: 1.85662
[2020-02-09 09:40:47,519 INFO] Validation accuracy: 90.6778
[2020-02-09 09:40:47,988 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_15000.pt
[2020-02-09 09:41:14,652 INFO] Step 15050/100000; acc:  99.07; ppl:  1.05; xent: 0.05; lr: 0.00072; 9990/10432 tok/s;   7932 sec
[2020-02-09 09:41:27,013 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 09:41:34,702 INFO] number of examples: 368702
[2020-02-09 09:41:50,961 INFO] Step 15100/100000; acc:  99.02; ppl:  1.06; xent: 0.05; lr: 0.00072; 10293/10769 tok/s;   7968 sec
[2020-02-09 09:42:15,404 INFO] Step 15150/100000; acc:  98.96; ppl:  1.06; xent: 0.06; lr: 0.00072; 15183/15944 tok/s;   7993 sec
[2020-02-09 09:42:39,853 INFO] Step 15200/100000; acc:  99.00; ppl:  1.06; xent: 0.05; lr: 0.00072; 15197/15941 tok/s;   8017 sec
[2020-02-09 09:43:04,353 INFO] Step 15250/100000; acc:  99.05; ppl:  1.06; xent: 0.05; lr: 0.00072; 15199/15932 tok/s;   8042 sec
[2020-02-09 09:43:28,888 INFO] Step 15300/100000; acc:  99.03; ppl:  1.06; xent: 0.05; lr: 0.00071; 15175/15911 tok/s;   8066 sec
[2020-02-09 09:43:53,380 INFO] Step 15350/100000; acc:  99.02; ppl:  1.06; xent: 0.05; lr: 0.00071; 15151/15912 tok/s;   8091 sec
[2020-02-09 09:44:17,531 INFO] Step 15400/100000; acc:  99.02; ppl:  1.06; xent: 0.05; lr: 0.00071; 15218/16064 tok/s;   8115 sec
[2020-02-09 09:44:41,852 INFO] Step 15450/100000; acc:  99.06; ppl:  1.05; xent: 0.05; lr: 0.00071; 15210/16004 tok/s;   8139 sec
[2020-02-09 09:45:06,462 INFO] Step 15500/100000; acc:  99.09; ppl:  1.05; xent: 0.05; lr: 0.00071; 15169/15878 tok/s;   8164 sec
[2020-02-09 09:45:06,463 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:45:06,566 INFO] number of examples: 10264
[2020-02-09 09:45:17,050 INFO] Validation perplexity: 1.98144
[2020-02-09 09:45:17,050 INFO] Validation accuracy: 90.3115
[2020-02-09 09:45:17,514 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_15500.pt
[2020-02-09 09:45:43,838 INFO] Step 15550/100000; acc:  99.08; ppl:  1.05; xent: 0.05; lr: 0.00071; 9883/10375 tok/s;   8201 sec
[2020-02-09 09:46:08,475 INFO] Step 15600/100000; acc:  99.10; ppl:  1.05; xent: 0.05; lr: 0.00071; 15205/15891 tok/s;   8226 sec
[2020-02-09 09:46:32,856 INFO] Step 15650/100000; acc:  99.04; ppl:  1.05; xent: 0.05; lr: 0.00071; 15204/15978 tok/s;   8250 sec
[2020-02-09 09:46:57,534 INFO] Step 15700/100000; acc:  99.07; ppl:  1.05; xent: 0.05; lr: 0.00071; 15160/15859 tok/s;   8275 sec
[2020-02-09 09:47:22,165 INFO] Step 15750/100000; acc:  99.08; ppl:  1.05; xent: 0.05; lr: 0.00070; 15201/15887 tok/s;   8299 sec
[2020-02-09 09:47:46,687 INFO] Step 15800/100000; acc:  99.11; ppl:  1.05; xent: 0.05; lr: 0.00070; 15171/15915 tok/s;   8324 sec
[2020-02-09 09:48:11,191 INFO] Step 15850/100000; acc:  99.03; ppl:  1.06; xent: 0.05; lr: 0.00070; 15076/15870 tok/s;   8349 sec
[2020-02-09 09:48:35,394 INFO] Step 15900/100000; acc:  99.11; ppl:  1.05; xent: 0.05; lr: 0.00070; 15233/16046 tok/s;   8373 sec
[2020-02-09 09:48:59,498 INFO] Step 15950/100000; acc:  99.04; ppl:  1.06; xent: 0.05; lr: 0.00070; 15271/16106 tok/s;   8397 sec
[2020-02-09 09:49:24,027 INFO] Step 16000/100000; acc:  99.08; ppl:  1.05; xent: 0.05; lr: 0.00070; 15178/15916 tok/s;   8421 sec
[2020-02-09 09:49:24,028 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:49:24,140 INFO] number of examples: 10264
[2020-02-09 09:49:34,479 INFO] Validation perplexity: 1.85019
[2020-02-09 09:49:34,480 INFO] Validation accuracy: 90.8058
[2020-02-09 09:49:34,946 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_16000.pt
[2020-02-09 09:50:01,544 INFO] Step 16050/100000; acc:  99.13; ppl:  1.05; xent: 0.05; lr: 0.00070; 9967/10428 tok/s;   8459 sec
[2020-02-09 09:50:16,400 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 09:50:22,510 INFO] number of examples: 368702
[2020-02-09 09:50:36,323 INFO] Step 16100/100000; acc:  99.12; ppl:  1.05; xent: 0.05; lr: 0.00070; 10754/11244 tok/s;   8494 sec
[2020-02-09 09:51:00,724 INFO] Step 16150/100000; acc:  99.09; ppl:  1.05; xent: 0.05; lr: 0.00070; 15183/15959 tok/s;   8518 sec
[2020-02-09 09:51:25,371 INFO] Step 16200/100000; acc:  99.14; ppl:  1.05; xent: 0.05; lr: 0.00069; 15158/15858 tok/s;   8543 sec
[2020-02-09 09:51:49,768 INFO] Step 16250/100000; acc:  99.15; ppl:  1.05; xent: 0.05; lr: 0.00069; 15186/15955 tok/s;   8567 sec
[2020-02-09 09:52:14,396 INFO] Step 16300/100000; acc:  99.18; ppl:  1.05; xent: 0.05; lr: 0.00069; 15150/15867 tok/s;   8592 sec
[2020-02-09 09:52:38,941 INFO] Step 16350/100000; acc:  99.16; ppl:  1.05; xent: 0.05; lr: 0.00069; 15111/15874 tok/s;   8616 sec
[2020-02-09 09:53:03,123 INFO] Step 16400/100000; acc:  99.12; ppl:  1.05; xent: 0.05; lr: 0.00069; 15210/16049 tok/s;   8640 sec
[2020-02-09 09:53:27,407 INFO] Step 16450/100000; acc:  99.14; ppl:  1.05; xent: 0.05; lr: 0.00069; 15196/16010 tok/s;   8665 sec
[2020-02-09 09:53:52,075 INFO] Step 16500/100000; acc:  99.18; ppl:  1.05; xent: 0.05; lr: 0.00069; 15154/15851 tok/s;   8689 sec
[2020-02-09 09:53:52,077 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:53:52,191 INFO] number of examples: 10264
[2020-02-09 09:54:02,570 INFO] Validation perplexity: 1.99544
[2020-02-09 09:54:02,571 INFO] Validation accuracy: 90.3547
[2020-02-09 09:54:03,039 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_16500.pt
[2020-02-09 09:54:29,398 INFO] Step 16550/100000; acc:  99.14; ppl:  1.05; xent: 0.05; lr: 0.00069; 9911/10397 tok/s;   8727 sec
[2020-02-09 09:54:53,982 INFO] Step 16600/100000; acc:  99.18; ppl:  1.05; xent: 0.05; lr: 0.00069; 15212/15912 tok/s;   8751 sec
[2020-02-09 09:55:18,461 INFO] Step 16650/100000; acc:  99.16; ppl:  1.05; xent: 0.05; lr: 0.00068; 15203/15948 tok/s;   8776 sec
[2020-02-09 09:55:43,094 INFO] Step 16700/100000; acc:  99.16; ppl:  1.05; xent: 0.05; lr: 0.00068; 15108/15843 tok/s;   8800 sec
[2020-02-09 09:56:07,764 INFO] Step 16750/100000; acc:  99.16; ppl:  1.05; xent: 0.05; lr: 0.00068; 15195/15872 tok/s;   8825 sec
[2020-02-09 09:56:32,259 INFO] Step 16800/100000; acc:  99.18; ppl:  1.05; xent: 0.05; lr: 0.00068; 15171/15924 tok/s;   8850 sec
[2020-02-09 09:56:56,845 INFO] Step 16850/100000; acc:  99.12; ppl:  1.05; xent: 0.05; lr: 0.00068; 15075/15842 tok/s;   8874 sec
[2020-02-09 09:57:21,058 INFO] Step 16900/100000; acc:  99.17; ppl:  1.05; xent: 0.05; lr: 0.00068; 15235/16043 tok/s;   8898 sec
[2020-02-09 09:57:45,200 INFO] Step 16950/100000; acc:  99.16; ppl:  1.05; xent: 0.05; lr: 0.00068; 15275/16094 tok/s;   8923 sec
[2020-02-09 09:58:09,661 INFO] Step 17000/100000; acc:  99.15; ppl:  1.05; xent: 0.05; lr: 0.00068; 15150/15925 tok/s;   8947 sec
[2020-02-09 09:58:09,663 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 09:58:09,774 INFO] number of examples: 10264
[2020-02-09 09:58:20,100 INFO] Validation perplexity: 1.86552
[2020-02-09 09:58:20,100 INFO] Validation accuracy: 90.7205
[2020-02-09 09:58:20,565 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_17000.pt
[2020-02-09 09:58:47,150 INFO] Step 17050/100000; acc:  99.22; ppl:  1.05; xent: 0.05; lr: 0.00068; 9986/10441 tok/s;   8984 sec
[2020-02-09 09:59:04,419 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 09:59:12,849 INFO] number of examples: 368702
[2020-02-09 09:59:24,214 INFO] Step 17100/100000; acc:  99.22; ppl:  1.05; xent: 0.05; lr: 0.00068; 10094/10552 tok/s;   9022 sec
[2020-02-09 09:59:48,515 INFO] Step 17150/100000; acc:  99.21; ppl:  1.05; xent: 0.05; lr: 0.00067; 15247/16026 tok/s;   9046 sec
[2020-02-09 10:00:13,237 INFO] Step 17200/100000; acc:  99.19; ppl:  1.05; xent: 0.05; lr: 0.00067; 15141/15825 tok/s;   9071 sec
[2020-02-09 10:00:37,564 INFO] Step 17250/100000; acc:  99.19; ppl:  1.05; xent: 0.05; lr: 0.00067; 15195/15984 tok/s;   9095 sec
[2020-02-09 10:01:02,103 INFO] Step 17300/100000; acc:  99.25; ppl:  1.05; xent: 0.05; lr: 0.00067; 15192/15917 tok/s;   9119 sec
[2020-02-09 10:01:26,637 INFO] Step 17350/100000; acc:  99.22; ppl:  1.05; xent: 0.05; lr: 0.00067; 15114/15880 tok/s;   9144 sec
[2020-02-09 10:01:50,929 INFO] Step 17400/100000; acc:  99.19; ppl:  1.05; xent: 0.05; lr: 0.00067; 15183/15997 tok/s;   9168 sec
[2020-02-09 10:02:15,064 INFO] Step 17450/100000; acc:  99.26; ppl:  1.05; xent: 0.05; lr: 0.00067; 15269/16099 tok/s;   9192 sec
[2020-02-09 10:02:39,631 INFO] Step 17500/100000; acc:  99.27; ppl:  1.05; xent: 0.04; lr: 0.00067; 15211/15917 tok/s;   9217 sec
[2020-02-09 10:02:39,633 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 10:02:39,736 INFO] number of examples: 10264
[2020-02-09 10:02:50,113 INFO] Validation perplexity: 1.9758
[2020-02-09 10:02:50,113 INFO] Validation accuracy: 90.3886
[2020-02-09 10:02:50,580 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_17500.pt
[2020-02-09 10:03:16,907 INFO] Step 17550/100000; acc:  99.25; ppl:  1.05; xent: 0.05; lr: 0.00067; 9924/10408 tok/s;   9254 sec
[2020-02-09 10:03:41,454 INFO] Step 17600/100000; acc:  99.24; ppl:  1.05; xent: 0.05; lr: 0.00067; 15217/15928 tok/s;   9279 sec
[2020-02-09 10:04:05,882 INFO] Step 17650/100000; acc:  99.25; ppl:  1.05; xent: 0.05; lr: 0.00067; 15234/15981 tok/s;   9303 sec
[2020-02-09 10:04:30,463 INFO] Step 17700/100000; acc:  99.27; ppl:  1.05; xent: 0.04; lr: 0.00066; 15151/15880 tok/s;   9328 sec
[2020-02-09 10:04:55,098 INFO] Step 17750/100000; acc:  99.24; ppl:  1.05; xent: 0.05; lr: 0.00066; 15214/15895 tok/s;   9352 sec
[2020-02-09 10:05:19,454 INFO] Step 17800/100000; acc:  99.28; ppl:  1.05; xent: 0.04; lr: 0.00066; 15225/15999 tok/s;   9377 sec
[2020-02-09 10:05:44,062 INFO] Step 17850/100000; acc:  99.20; ppl:  1.05; xent: 0.05; lr: 0.00066; 15084/15839 tok/s;   9401 sec
[2020-02-09 10:06:08,238 INFO] Step 17900/100000; acc:  99.26; ppl:  1.05; xent: 0.05; lr: 0.00066; 15259/16069 tok/s;   9426 sec
[2020-02-09 10:06:32,347 INFO] Step 17950/100000; acc:  99.22; ppl:  1.05; xent: 0.05; lr: 0.00066; 15287/16111 tok/s;   9450 sec
[2020-02-09 10:06:56,839 INFO] Step 18000/100000; acc:  99.12; ppl:  1.05; xent: 0.05; lr: 0.00066; 15179/15929 tok/s;   9474 sec
[2020-02-09 10:06:56,841 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 10:06:56,952 INFO] number of examples: 10264
[2020-02-09 10:07:07,270 INFO] Validation perplexity: 1.98865
[2020-02-09 10:07:07,270 INFO] Validation accuracy: 90.3476
[2020-02-09 10:07:07,735 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_18000.pt
[2020-02-09 10:07:34,292 INFO] Step 18050/100000; acc:  99.22; ppl:  1.05; xent: 0.05; lr: 0.00066; 9976/10441 tok/s;   9512 sec
[2020-02-09 10:07:54,056 INFO] Loading dataset from data/demo.train.0.pt
[2020-02-09 10:08:01,791 INFO] number of examples: 368702
[2020-02-09 10:08:10,735 INFO] Step 18100/100000; acc:  99.28; ppl:  1.04; xent: 0.04; lr: 0.00066; 10264/10731 tok/s;   9548 sec
[2020-02-09 10:08:35,098 INFO] Step 18150/100000; acc:  99.26; ppl:  1.05; xent: 0.04; lr: 0.00066; 15231/15997 tok/s;   9572 sec
[2020-02-09 10:08:59,735 INFO] Step 18200/100000; acc:  99.27; ppl:  1.05; xent: 0.04; lr: 0.00066; 15152/15861 tok/s;   9597 sec
[2020-02-09 10:09:24,050 INFO] Step 18250/100000; acc:  99.28; ppl:  1.05; xent: 0.04; lr: 0.00065; 15197/15988 tok/s;   9621 sec
[2020-02-09 10:09:48,732 INFO] Step 18300/100000; acc:  99.30; ppl:  1.04; xent: 0.04; lr: 0.00065; 15169/15861 tok/s;   9646 sec
[2020-02-09 10:10:13,256 INFO] Step 18350/100000; acc:  99.29; ppl:  1.04; xent: 0.04; lr: 0.00065; 15111/15879 tok/s;   9671 sec
[2020-02-09 10:10:37,643 INFO] Step 18400/100000; acc:  99.29; ppl:  1.04; xent: 0.04; lr: 0.00065; 15142/15944 tok/s;   9695 sec
[2020-02-09 10:11:01,767 INFO] Step 18450/100000; acc:  99.32; ppl:  1.04; xent: 0.04; lr: 0.00065; 15264/16100 tok/s;   9719 sec
[2020-02-09 10:11:26,255 INFO] Step 18500/100000; acc:  99.33; ppl:  1.04; xent: 0.04; lr: 0.00065; 15194/15936 tok/s;   9744 sec
[2020-02-09 10:11:26,257 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 10:11:26,365 INFO] number of examples: 10264
[2020-02-09 10:11:36,719 INFO] Validation perplexity: 1.94439
[2020-02-09 10:11:36,719 INFO] Validation accuracy: 90.4273
[2020-02-09 10:11:37,187 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_18500.pt
[2020-02-09 10:12:03,590 INFO] Step 18550/100000; acc:  99.31; ppl:  1.04; xent: 0.04; lr: 0.00065; 9928/10402 tok/s;   9781 sec
[2020-02-09 10:12:28,136 INFO] Step 18600/100000; acc:  99.32; ppl:  1.04; xent: 0.04; lr: 0.00065; 15220/15929 tok/s;   9805 sec
[2020-02-09 10:12:52,676 INFO] Step 18650/100000; acc:  99.32; ppl:  1.04; xent: 0.04; lr: 0.00065; 15203/15927 tok/s;   9830 sec
[2020-02-09 10:13:17,133 INFO] Step 18700/100000; acc:  99.30; ppl:  1.04; xent: 0.04; lr: 0.00065; 15179/15936 tok/s;   9854 sec
[2020-02-09 10:13:41,685 INFO] Step 18750/100000; acc:  99.30; ppl:  1.04; xent: 0.04; lr: 0.00065; 15234/15933 tok/s;   9879 sec
[2020-02-09 10:14:06,201 INFO] Step 18800/100000; acc:  99.31; ppl:  1.04; xent: 0.04; lr: 0.00064; 15190/15927 tok/s;   9904 sec
[2020-02-09 10:14:30,714 INFO] Step 18850/100000; acc:  99.28; ppl:  1.05; xent: 0.04; lr: 0.00064; 15143/15901 tok/s;   9928 sec
[2020-02-09 10:14:54,946 INFO] Step 18900/100000; acc:  99.32; ppl:  1.04; xent: 0.04; lr: 0.00064; 15203/16021 tok/s;   9952 sec
[2020-02-09 10:15:19,069 INFO] Step 18950/100000; acc:  99.34; ppl:  1.04; xent: 0.04; lr: 0.00064; 15279/16103 tok/s;   9976 sec
[2020-02-09 10:15:43,660 INFO] Step 19000/100000; acc:  99.33; ppl:  1.04; xent: 0.04; lr: 0.00064; 15143/15876 tok/s;  10001 sec
[2020-02-09 10:15:43,662 INFO] Loading dataset from data/demo.valid.0.pt
[2020-02-09 10:15:43,773 INFO] number of examples: 10264
[2020-02-09 10:15:54,084 INFO] Validation perplexity: 2.03557
[2020-02-09 10:15:54,085 INFO] Validation accuracy: 90.3098
[2020-02-09 10:15:54,558 INFO] Saving checkpoint models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_19000.pt
[2020-02-09 10:16:20,998 INFO] Step 19050/100000; acc:  99.35; ppl:  1.04; xent: 0.04; lr: 0.00064; 9976/10459 tok/s;  10038 sec
^CTraceback (most recent call last):
  File "./src/train.py", line 6, in <module>