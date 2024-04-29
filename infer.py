import os
import torch

import config

from data import load_tokenizer, create_dataloader
from model import build_model

def test():
    tokenizer = load_tokenizer()
    model = build_model(config.vocab_size, config.d_model, \
        config.n_heads, config.n_layers, config.d_ff, config.dropout)
    
    file_path = "%sfinal.pt" % config.file_prefix
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
    model.eval()

    test_dataloader, test_size = create_dataloader(config.test_file, \
        1, config.max_padding, shuffle=False, drop_last=False)

    print("Testing ====")
    for i, testcase in enumerate(test_dataloader):
        print(f"Case {i+1} / {test_size}", flush=True)
        tokens = testcase.tokens[0]
        labels = testcase.labels[0]
        for j, label in enumerate(labels):
            if label:
                tokens[j] = label
        print('Origin:', testcase.sents[0])
        print('Is Next:', testcase.is_next)
        print('Masked:', tokenizer.DecodeIds(tokens.tolist()))
        print('Output:', tokenizer.DecodeIds(tokens.tolist()))

if __name__ == '__main__':
    test()
