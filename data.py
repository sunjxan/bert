import torch
import sentencepiece as spm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import config

DEVICE = config.device

def load_tokenizers():
    src_model_path = config.src_model_prefix + '.model'
    tokenizer_src = spm.SentencePieceProcessor()
    tokenizer_src.Load(src_model_path)

    tgt_model_path = config.tgt_model_prefix + '.model'
    tokenizer_tgt = spm.SentencePieceProcessor()
    tokenizer_tgt.Load(tgt_model_path)
    
    return tokenizer_src, tokenizer_tgt

class MyDataset(Dataset):
    def __init__(self, src_input_file, tgt_input_file):
        super().__init__()
        with open(src_input_file, 'r', encoding='UTF8') as f:
            self._src_texts = list(map(lambda x: x.strip(), f))
        with open(tgt_input_file, 'r', encoding='UTF8') as f:
            self._tgt_texts = list(map(lambda x: x.strip(), f))
        self._size = min(len(self._src_texts), len(self._tgt_texts))

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if index >= 0 and index < self.__len__():
            src_sent = self._src_texts[index]
            tgt_sent = self._tgt_texts[index]
            return src_sent, tgt_sent
        return

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

class Batch:
    def __init__(self, src, tgt=None, src_pad=0, tgt_pad=0, src_texts=None, tgt_texts=None):
        # src (N,S)
        # src_mask (N,1,S)
        # tgt, tgt_y (N,T-1)
        # tgt_mask (N,T-1,T-1)
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src = src
        self.src_mask = (src != src_pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, tgt_pad)
            self.ntokens = (self.tgt_y != tgt_pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

def collate_batch(batch, tokenizer_src, tokenizer_tgt, max_padding=128):
    # 在每个句子前后加上BOS和EOS，再进行填充截断处理，得到(N,S)的src和(N,T)的tgt，创建Batch对象
    src_texts, tgt_texts = [], []
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                torch.tensor([tokenizer_src.bos_id()], device=DEVICE),
                torch.tensor(
                    tokenizer_src.EncodeAsIds(_src)[:max_padding], dtype=torch.int64, device=DEVICE
                ),
                torch.tensor([tokenizer_src.eos_id()], device=DEVICE)
            ], dim=0
        )
        processed_tgt = torch.cat(
            [
                torch.tensor([tokenizer_tgt.bos_id()], device=DEVICE),
                torch.tensor(
                    tokenizer_tgt.EncodeAsIds(_tgt)[:max_padding], dtype=torch.int64, device=DEVICE
                ),
                torch.tensor([tokenizer_tgt.eos_id()], device=DEVICE)
            ], dim=0
        )
        src_list.append(processed_src)
        tgt_list.append(processed_tgt)
        src_texts.append(_src)
        tgt_texts.append(_tgt)
    src_pad = tokenizer_src.pad_id()
    tgt_pad = tokenizer_tgt.pad_id()
    src = pad_sequence(src_list, batch_first=True, padding_value=src_pad)
    tgt = pad_sequence(tgt_list, batch_first=True, padding_value=tgt_pad)
    return Batch(src, tgt, src_pad, tgt_pad, src_texts, tgt_texts)

def create_dataloader(src_input_file, tgt_input_file, batch_size, max_padding=128, shuffle=False, drop_last=False):
    dataset = MyDataset(src_input_file, tgt_input_file)
    tokenizer_src, tokenizer_tgt = load_tokenizers()

    def collate_fn(batch):
        return collate_batch(batch, tokenizer_src, tokenizer_tgt, max_padding)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn)
    return dataloader
