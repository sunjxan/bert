import math
import random
import torch
import sentencepiece as spm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import config

DEVICE = config.device

def load_tokenizer():
    model_path = config.model_prefix + '.model'
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(model_path)
    
    return tokenizer

class MyDataset(Dataset):
    def __init__(self, input_file):
        super().__init__()
        with open(input_file, 'r', encoding='UTF8') as f:
            self._sents = list(map(lambda x: x.strip(), f))
        self._size = len(self._sents)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if index >= 0 and index < self.__len__():
            sent = self._sents[index]
            return sent, index
        return

class Batch:
    def __init__(self, tokens, labels, segments, is_next, pad_id=0):
        self.tokens = tokens
        self.labels = labels
        self.segments = segments
        self.is_next = is_next
        self.mask = (self.tokens != pad_id).unsqueeze(-2)

def mask_word(sent, tokenizer, vocab_size, pad=0):
    tokens = tokenizer.EncodeAsIds(sent)
    labels = []

    for i, token in enumerate(tokens):
        prob = random.random()
        if prob < .15:
            prob /= .15
            if prob < .8:
                tokens[i] = pad
            elif prob <.9:
                tokens[i] = random.choice(range(vocab_size))
            labels.append(token)
        else:
            labels.append(0)
    return tokens, labels

def collate_batch(batch, tokenizer, dataset, max_padding=256):
    token_list, label_list, segment_list, is_next_list = [], [], [], []

    size = len(dataset)
    pad_id = tokenizer.pad_id()
    for (first, index) in batch:
        if index == size - 1:
            second, _ = random.choice(dataset)
            is_next = False
        elif random.random() > .5:
            second, _ = dataset[index + 1]
            is_next = True
        else:
            second, ix = random.choice(dataset)
            is_next = (ix == index + 1)
        first_tokens, first_labels = mask_word(first, tokenizer, config.vocab_size, pad_id)
        second_tokens, second_labels = mask_word(second, tokenizer, config.vocab_size, pad_id)
        first_segment = [1] * (len(first_tokens) + 2)
        second_segment = [2] * (len(second_tokens) + 1)

        processed_tokens = torch.cat(
            [
                torch.tensor([tokenizer.bos_id()], device=DEVICE),
                torch.tensor(first_tokens, dtype=torch.int64, device=DEVICE),
                torch.tensor([tokenizer.eos_id()], device=DEVICE),
                torch.tensor(second_tokens, dtype=torch.int64, device=DEVICE),
                torch.tensor([tokenizer.eos_id()], device=DEVICE)
            ], dim=0
        )[:max_padding]
        processed_labels = torch.cat(
            [
                torch.tensor([pad_id], device=DEVICE),
                torch.tensor(first_labels, dtype=torch.int64, device=DEVICE),
                torch.tensor([pad_id], device=DEVICE),
                torch.tensor(second_labels, dtype=torch.int64, device=DEVICE),
                torch.tensor([pad_id], device=DEVICE)
            ], dim=0
        )[:max_padding]
        segment_labels = torch.cat(
            [
                torch.tensor(first_segment, dtype=torch.int64, device=DEVICE),
                torch.tensor(second_segment, dtype=torch.int64, device=DEVICE)
            ], dim=0
        )[:max_padding]

        token_list.append(processed_tokens)
        label_list.append(processed_labels)
        segment_list.append(segment_labels)
        is_next_list.append([is_next])
    
    tokens = pad_sequence(token_list, batch_first=True, padding_value=pad_id)
    labels = pad_sequence(label_list, batch_first=True, padding_value=pad_id)
    segments = pad_sequence(segment_list, batch_first=True, padding_value=pad_id)
    is_next = torch.tensor(is_next_list, dtype=torch.int64, device=DEVICE)
    return Batch(tokens, labels, segments, is_next, pad_id)

def create_dataloader(input_file, batch_size, max_padding=256, shuffle=False, drop_last=False):
    dataset = MyDataset(input_file)
    tokenizer = load_tokenizer()

    def collate_fn(batch):
        return collate_batch(batch, tokenizer, dataset, max_padding)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn)
    if drop_last:
        size = math.floor(len(dataset) / batch_size)
    else:
        size = math.ceil(len(dataset) / batch_size)
    return dataloader, size
