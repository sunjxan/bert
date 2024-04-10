import os
import torch

import config

from data import load_tokenizers, create_dataloader, subsequent_mask
from model import build_transformer

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol=None):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        if next_word == end_symbol:
            return ys
    return ys

def test():
    tokenizer_src, tokenizer_tgt = load_tokenizers()
    model = build_transformer(config.src_vocab_size, config.tgt_vocab_size, \
        config.d_model, config.n_heads, config.n_layers, config.d_ff, config.dropout)
    
    file_path = "%sfinal.pt" % config.file_prefix
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
    model.eval()

    test_dataloader = create_dataloader(config.src_test_file, config.tgt_test_file, \
        1, config.max_padding, shuffle=False, drop_last=False)
    
    for i, testcase in enumerate(test_dataloader):
        print(f"Case {i} Test ====", flush=True)
        res = greedy_decode(model, testcase.src, testcase.src_mask, max_len=20, \
            start_symbol=tokenizer_tgt.bos_id(), end_symbol=tokenizer_tgt.eos_id())
        print(tokenizer_tgt.DecodeIds(res[0]))

if __name__ == '__main__':
    test()
