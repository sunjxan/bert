from torch.utils.data import Dataset, DataLoader

import config

DEVICE = config.device

def load_tokenizers():
    src_model_path = config.src_model_prefix + '.model'
    tokenizer_src = spm.SentenPieceProcessor()
    tokenizer_src.Load(src_model_path)

    tgt_model_path = config.tgt_model_prefix + '.model'
    tokenizer_tgt = spm.SentenPieceProcessor()
    tokenizer_tgt.Load(tgt_model_path)
    
    return tokenizer_src, tokenizer_tgt

class MyDataset(Dataset):
    def __init__(self, src_input_file, tgt_input_file):
        super().__init__()
        with open(src_input_file, 'r', encoding='UTF8') as f:
            self._src_sents = list(map(lambda x: x.strip(), f))
        with open(tgt_input_file, 'r', encoding='UTF8') as f:
            self._tgt_sents = list(map(lambda x: x.strip(), f))
        self._size = min(len(self._src_sents), len(self._tgt_sents))

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if index >= 0 and index < self.__len__():
            src_sent = self._src_sents[index]
            tgt_sent = self._tgt_sents[index]
            return src_sent, tgt_sent
        return

class Batch:
    def __init__(self):
        pass

def collate_batch(batch):
    tokenizer_src, tokenizer_tgt = load_tokenizers()

def create_dataloader(src_input_file, tgt_input_file, batch_size, shuffle=False, drop_last=False):
    dataset = MyDataset(src_input_file, tgt_input_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_batch)
    return dataloader
