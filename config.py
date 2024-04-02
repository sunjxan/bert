src_input_file = './data/corpus_eng.txt'
src_vocab_size = 32000
src_model_prefix = 'eng'
src_model_type = 'bpe'
src_character_coverage = 1

tgt_input_file = './data/corpus_chn.txt'
tgt_vocab_size = 32000
tgt_model_prefix = 'chn'
tgt_model_type = 'bpe'
tgt_character_coverage = .9995

padding_id = 0
unknown_id = 1
bos_id = 2
eos_id = 3
