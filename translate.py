import logging
import sys
import json
from dataclasses import dataclass

from onmt.translate.translator import build_translator
from onmt.utils.parse import ArgumentParser

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("model")


@dataclass
class ModelOpts:
    alpha = 0.0
    attn_debug = False
    avg_raw_probs = False
    batch_size = 30
    beam_size = 5
    beta = -0.0
    block_ngram_repeat = 0
    config = None
    coverage_penalty = 'none'
    data_type = 'text'
    dump_beam = ''
    dynamic_dict = False
    fp32 = False
    gpu = -1
    ignore_when_blocking = []
    image_channel_size = 3
    length_penalty = 'none'
    log_file = ''
    log_file_level = '0'
    max_length = 100
    max_sent_length = None
    min_length = 0
    n_best = 1
    output = 'pred.txt'
    phrase_table = ''
    random_sampling_temp = 1.0
    random_sampling_topk = 1
    ratio = -0.0
    replace_unk = True
    report_bleu = False
    report_rouge = False
    report_time = False
    sample_rate = 16000
    save_config = None
    seed = 829
    shard_size = 10000
    share_vocab = False
    src = 'one_phrase.txt'
    src_dir = ''
    stepwise_penalty = False
    tgt = None
    models = None
    verbose = True
    window = 'hamming'
    window_size = 0.02
    window_stride = 0.01
    sm_data_path = "/.sagemaker/mms/models/model/"


opt = ModelOpts()

ArgumentParser.validate_translate_opts(opt)


def model_fn(model_dir):
    opt.models = model_dir
    return build_translator(opt, report_score=True)


def input_fn(request_body, request_content_type):
    data = request_body.get('text')
    return json.loads(data.encode('ascii'))


def predict_fn(input_data, model):
    src_shard = json.loads(input_data)
    tgt_shard = None

    score, prediction = model.translate(
        src=src_shard,
        tg=tgt_shard,
        src_dir=opt.src_dir,
        batch_size=opt.batch_size,
        attn_debug=opt.attn_debug

    )

    translation = {
        "score": score,
        "prediction_data": prediction,
        "prediction": prediction[0][0]
    }
    return translation


def output_fn(prediction, content_type):
    return prediction
