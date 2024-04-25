import torch

train_file = './data/train/corpus.zh'
val_file = './data/val/corpus.zh'
test_file = './data/test/corpus.zh'
vocab_size = 32000
model_prefix = 'zh'
model_type = 'bpe'
character_coverage = .9995

max_padding = 256
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
print_iter = 40
file_prefix = 'model_'

device = torch.device('cpu')