import os
import sentencepiece as spm

import config

def train(input_file, model_prefix, vocab_size, model_type, character_coverage):
    arg = f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} \
--model_type={model_type} --character_coverage={character_coverage} \
--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
    spm.SentencePieceTrainer.Train(arg)

if __name__ == '__main__':
    train(config.src_input_file, config.src_model_prefix, config.src_vocab_size, config.src_model_type, config.src_character_coverage)
    train(config.tgt_input_file, config.tgt_model_prefix, config.tgt_vocab_size, config.tgt_model_type, config.tgt_character_coverage)
