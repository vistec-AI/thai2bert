import math
from typing import List, Dict, Union, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

SPECIAL_TOKEN_NAMES = ['bos_token', 'eos_token',
                       'sep_token', 'cls_token', 'pad_token']


@dataclass
class DataCollatorForSpanLevelMask(DataCollatorForLanguageModeling):
    """
    Data collator used for span-level masked language modeling

    adapted from NGramMaskGenerator class

    https://github.com/microsoft/DeBERTa/blob/11fa20141d9700ba2272b38f2d5fce33d981438b/DeBERTa/apps/tasks/mlm_task.py#L36
    and
    https://github.com/zihangdai/xlnet/blob/0b642d14dd8aec7f1e1ecbf7d6942d5faa6be1f0/data_utils.py

    """
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    max_gram: int = 3
    pad_to_multiple_of: Optional[int] = None
    max_preds_per_seq: int = None
    max_seq_len: int = 510

    def __new__(cls, tokenizer, mlm, mlm_probability, pad_to_multiple_of, *args, **kwargs):

        obj = object.__new__(cls)
        DataCollatorForLanguageModeling.__init__(obj, tokenizer=tokenizer, mlm=mlm,
                                                 mlm_probability=mlm_probability,
                                                 pad_to_multiple_of=pad_to_multiple_of)
        return obj

    def __post_init__(self, *args, **kwargs):

        if self.max_preds_per_seq is None:
            self.max_preds_per_seq = math.ceil(self.max_seq_len * self.mlm_probability / 10) * 10 # make ngrams per window sized context
            self.mask_window = torch.FloatTensor([float(1 / self.mlm_probability)])
        self.vocab_words = list(self.tokenizer.get_vocab().keys())
        self.vocab_mapping = self.tokenizer.get_vocab()

        self.special_token_ids = [self.vocab_mapping[self.tokenizer.special_tokens_map[name]] for name in SPECIAL_TOKEN_NAMES]
        self.ngrams = np.arange(1, self.max_gram + 1, dtype=np.int64)

        _pvals = 1. / np.arange(1, self.max_gram + 1)
        self.pvals = torch.Tensor(_pvals / _pvals.sum(keepdims=True))

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        SEQ_LEN = torch.LongTensor([inputs.shape[1]])

        labels = inputs.clone()
        masked_indices = torch.zeros(inputs.shape).bool()
        _masked_indices_rand = torch.rand(inputs.shape)
        if special_tokens_mask is None:
            special_tokens_mask = sum(inputs == i for i in self.special_token_ids).bool()
        else:
            special_tokens_mask = special_tokens_mask.bool()

        _masked_indices_rand.masked_fill_(special_tokens_mask, value=-1.0)

        offset = torch.LongTensor([0])
        c = 0
        while offset < SEQ_LEN:

            n = torch.FloatTensor([torch.multinomial(self.pvals, 1, replacement=True) + 1])
            ctx_size = torch.min(torch.ceil( n * self.mask_window).type(torch.LongTensor), SEQ_LEN - offset)
            
            if c == 0:
                _sub_masked_indices = masked_indices[:, offset+1: offset+1+ctx_size]
                _sub_masked_indices_rand = _masked_indices_rand[:, offset+1: offset+1+ctx_size]
            else:
                _sub_masked_indices = masked_indices[:, offset: offset+ctx_size]
                _sub_masked_indices_rand = _masked_indices_rand[:, offset: offset+ctx_size]


            start = torch.argmax(_sub_masked_indices_rand, dim=-1)
            end = torch.min(start + n, ctx_size)

            indices = torch.stack((start, end), dim=-1)

            for i, ind in enumerate(indices):
                _sub_masked_indices[i, ind[0]:ind[1]] = True

            if c == 0:
                offset += ctx_size + 1
            else:
                offset += ctx_size
            c += 1

        masked_indices[special_tokens_mask] = False

        labels[~masked_indices] = -100
        inputs[masked_indices] = self.tokenizer.mask_token_id

        return inputs, labels
