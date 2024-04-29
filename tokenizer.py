import os
import sentencepiece as spm

import config

def train(input_file, model_prefix, vocab_size, model_type, character_coverage):
    arg = f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size-1} \
--model_type={model_type} --character_coverage={character_coverage} \
--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
    spm.SentencePieceTrainer.Train(arg)

if __name__ == '__main__':
    train(config.train_file, config.model_prefix, config.vocab_size, config.model_type, config.character_coverage)
