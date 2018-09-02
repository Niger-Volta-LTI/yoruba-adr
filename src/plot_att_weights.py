import onmt
import onmt.io
import onmt.translate
import onmt.ModelConstructor
from collections import namedtuple
import numpy as np

# Load the model.
Opt = namedtuple('Opt', ['model', 'data_type', 'reuse_copy_attn', "gpu"])

opt = Opt("/home/ubuntu/github/OpenNMT-py/models/saves/yo_adr_bahdanau_lstm_256_1_1_acc_89.83_ppl_1.69_e11.pt", "text",False, 0)
fields, model, model_opt =  onmt.ModelConstructor.load_test_model(opt,{"reuse_copy_attn":False})

# Test data
data = onmt.io.build_dataset(fields, "text", "/home/ubuntu/github/OpenNMT-py/data/src-val.txt", None, use_filter_pred=False)
data_iter = onmt.io.OrderedIterator(
        dataset=data, device=0,
        batch_size=1, train=False, sort=False,
        sort_within_batch=True, shuffle=False)
# Translator
translator = onmt.translate.Translator(model, fields,
                                           beam_size=5,
                                           n_best=1,
                                           global_scorer=onmt.translate.GNMTGlobalScorer(0, 0, "none", "none"),
                                           cuda=True)

builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        1, False, None)

for j, batch in enumerate(data_iter):
        batch_data = translator.translate_batch(batch, data)
        translations = builder.from_batch(batch_data)
        print("src:", " ".join(translations[0].src_raw))
        print("tgt:", " ".join(translations[0].pred_sents[0]))
        translations[0].log(j)
        np.save("./toprint/" + "_".join(translations[0].src_raw) + ".npy", (translations[0].attns[0]).cpu().numpy())
        print()

