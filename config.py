import torch

src_train_file = './data/train/corpus.en'
src_eval_file = './data/eval/corpus.en'
src_test_file = './data/test/corpus.en'
src_vocab_size = 32000
src_model_prefix = 'en'
src_model_type = 'bpe'
src_character_coverage = 1

tgt_train_file = './data/train/corpus.zh'
tgt_eval_file = './data/eval/corpus.zh'
tgt_test_file = './data/test/corpus.zh'
tgt_vocab_size = 32000
tgt_model_prefix = 'zh'
tgt_model_type = 'bpe'
tgt_character_coverage = .9995

max_padding = 128
batch_size = 32

d_model = 512
n_heads = 8
n_layers = 6
d_ff = 2048
dropout = .1

base_lr = 1.0
warmup = 3000
num_epochs = 8
accum_iter = 10

device = torch.device('cpu')