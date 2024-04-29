import os
import torch

import config

from data import load_tokenizer, create_dataloader
from model import build_model

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
        print('Src:', testcase.src_sents[0])
        print('Target:', testcase.tgt_sents[0])
        res = greedy_decode(model, testcase.src, testcase.src_mask, max_len=50, \
            start_symbol=tokenizer_tgt.bos_id(), end_symbol=tokenizer_tgt.eos_id())
        print('Output:', tokenizer_tgt.DecodeIds(res[0].tolist()))

if __name__ == '__main__':
    test()
