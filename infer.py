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
        print('Origin:', testcase.sents[0])
        print('Is Next:', bool(testcase.is_next[0]))
        tokens = testcase.tokens[0]
        labels = testcase.labels[0]
        masked = ''
        pos = 0
        for j, label in enumerate(labels):
            if label:
                if j > pos:
                    masked += tokenizer.DecodeIds(tokens[pos:j].tolist())
                else:
                    masked += '<mask>'
                pos = j + 1
        if tokens.size(0) > pos:
            masked += tokenizer.DecodeIds(tokens[pos:].tolist())
        else:
            masked += '<mask>'
        print('Masked:', masked)
        mask_lm_out, next_sent_out = model.forward(testcase.tokens, testcase.segments, testcase.mask)
        print(mask_lm_out.shape, next_sent_out.shape)
        # print('Output:', tokenizer.DecodeIds(tokens.tolist()))

if __name__ == '__main__':
    test()
