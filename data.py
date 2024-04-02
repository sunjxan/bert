from torch.utils.data import Dataset

import config

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open(config.src_input_file, 'r', encoding='UTF8') as f:
            self._src_sents = list(map(lambda x: x.strip(), f))
        with open(config.tgt_input_file, 'r', encoding='UTF8') as f:
            self._tgt_sents = list(map(lambda x: x.strip(), f))

        sorted_index = sorted(range(len(src_sents)), key=lambda i: len(src_sents[i]))
        self._src_sents = [self._src_sents[i] for i in sorted_index]
        self._tgt_sents = [self._tgt_sents[i] for i in sorted_index]
        self._size = min(len(self._src_sents), len(self._tgt_sents))

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if index >= 0 and index < self.__len__():
            src_sent = self._src_sents[index]
            tgt_sent = self._tgt_sents[index]
            return src_sent, tgt_sent
        return
        