import sentencepiece as spm

import config

def train(input_file, model_prefix, vocab_size, model_type, character_coverage):
    arg = f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} \
--model_type={model_type} --character_coverage={character_coverage} \
--pad_id={config.padding_id} --unk_id={config.unknown_id} \
--bos_id={config.bos_id} --eos_id={config.eos_id}'
    spm.SentencePieceTrainer.Train(arg)

if __name__ == '__main__':
    train(config.src_input_file, config.src_model_prefix, config.src_vocab_size, 'bpe', 1)
    train(config.tgt_input_file, config.tgt_model_prefix, config.tgt_vocab_size, 'bpe', .9995)
