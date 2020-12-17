# Training tokenizer

This step train a tokenizer (output a vocaulary file) so it can be use later. Currently, there are multiple tokenizer that can be trained using `scripts/train_tokenizer.py`. This all done by cutting up text using predefined model and make a vocabulary with specified constrain, such as minimum frequency of word found in dataset.

## Type of tokenizer

 1. newmm - Dictionary-based word-level maximal matching tokenizer from [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp)
 2. syllable - cut by word found in syllable corpus.
 3. fake_sefr_cut - cut word by `SEFR_SPLIT_TOKEN` or in its current form `<|>`.

## How to

The following command can be use to train a tokenizer. We can also use `--help` to get more information.

```bash
python3 train_tokenizer.py \
 --ext txt \
 --train_dir "$PROJECT_TRAIN_DATASET_DIR" \
 --eval_dir "$PROJECT_EVAL_DATASET_DIR" \
 --output_file "$PROJECT_TOKENIZER_PATH/$PROJECT_PRE_TOKENIZER_TYPE.json" \
 --pre_tokenizer_type "$PROJECT_PRE_TOKENIZER_TYPE" \
 --overwrite_output_file \
 --vocab_min_freq "$PROJECT_VOCAB_MIN_FREQ"
```

The command above will read `*.txt` file in `$PROJECT_TRAIN_DATASET_DIR` line by line, strip, and ignore empty line. Then cut the each line in to multiple words and count words occurance. Filter out words, which has less frequency than `$PROJECT_VOCAB_MIN_FREQ`. Then dump the words and their coresponding ids into `$PROJECT_TOKENIZER_PATH/$PROJECT_PRE_TOKENIZER_TYPE.json`.