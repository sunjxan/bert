import os
import sentencepiece as spm

import config

def train(input_file, model_prefix, vocab_size, model_type, character_coverage):
    arg = f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} \
--model_type={model_type} --character_coverage={character_coverage} \
--pad_id={config.padding_id} --unk_id={config.unknown_id} \
--bos_id={config.bos_id} --eos_id={config.eos_id}'
    spm.SentencePieceTrainer.Train(arg)

def load_tokenizers():
    src_model_path = config.src_model_prefix + '.model'
    if not os.path.exists(src_model_path):
        train(config.src_input_file, config.src_model_prefix, config.src_vocab_size, config.src_model_type, config.src_character_coverage)
    src_sp = spm.SentenPieceProcessor()
    src_sp.Load(src_model_path)

    tgt_model_path = config.tgt_model_prefix + '.model'
    if not os.path.exists(tgt_model_path):
        train(config.tgt_input_file, config.tgt_model_prefix, config.tgt_vocab_size, config.tgt_model_type, config.tgt_character_coverage)
    tgt_sp = spm.SentenPieceProcessor()
    tgt_sp.Load(tgt_model_path)
    
    return src_sp, tgt_sp
