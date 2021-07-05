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

    def __new__(cls, tokenizer, mlm, mlm_probability, pad_to_multiple_of, *args, **kwargs):
    
        obj = object.__new__(cls)
        DataCollatorForLanguageModeling.__init__(obj, tokenizer=tokenizer, mlm=mlm,
                                                 mlm_probability=mlm_probability,
                                                 pad_to_multiple_of=pad_to_multiple_of)
        return obj
    

    def __post_init__(self, *args, **kwargs):
        
        self.vocab_words = list(self.tokenizer.get_vocab().keys())
        self.vocab_mapping = self.tokenizer.get_vocab()
        
        self.special_token_ids = [ self.vocab_mapping[self.tokenizer.special_tokens_map[name]] for name in  SPECIAL_TOKEN_NAMES]
        self.ngrams = np.arange(1, self.max_gram + 1, dtype=np.int64)
        _pvals = 1. / np.arange(1, self.max_gram + 1)
        self.pvals = torch.Tensor(_pvals / _pvals.sum(keepdims=True))

    def filter_indices(self, base_indices, to_be_filtered_indices):      

        to_filter_indices = to_be_filtered_indices[:,1].tolist()
        
        keep_indices = list(set(range(base_indices.shape[0])).difference(set(to_filter_indices)))

        base_indices_filtered = torch.index_select(base_indices, dim=0, index=torch.LongTensor(keep_indices))

        if len(to_filter_indices) == 0:
            return base_indices_filtered, None
        
        base_indices_selected = torch.index_select(base_indices, dim=0, index=torch.LongTensor(to_filter_indices))

        return base_indices_filtered, base_indices_selected


    def mask_tokens(self, inputs: torch.Tensor,
                    special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        labels = inputs.clone()
        labels_to_be_mask = torch.full(inputs.shape, 0., dtype=torch.bool)
           
        if special_tokens_mask is None:
            special_tokens_mask = sum(inputs==i for i in self.special_token_ids).bool()
        else:
            special_tokens_mask = special_tokens_mask.bool()

        K = len(self.pvals)
        mask_indices_by_span_len = [[] for i in range(K)]

        probability_matrix = torch.full(inputs.shape, self.mlm_probability, dtype=torch.float)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        base_masked_indices = torch.bernoulli(probability_matrix).bool()

        base_indices = (base_masked_indices == True).nonzero(as_tuple=False)


        _filter_base_indices = base_indices.clone()
        for k in range(0, K):

            _probabilty_matrix = torch.full((1, _filter_base_indices.shape[0]), self.pvals[k], dtype=torch.float)

            _masked_indices = torch.bernoulli(_probabilty_matrix).bool()

            to_be_filtered_indices = (_masked_indices == True).nonzero(as_tuple=False)

            _filter_base_indices, _indices_selected = self.filter_indices(_filter_base_indices, to_be_filtered_indices)

            if _indices_selected == None:
                mask_indices_by_span_len[k] = torch.LongTensor([])
            else:
                mask_indices_by_span_len[k] = _indices_selected
        
        
        # Applying span-level masking
        accum_indices = [[],[]]
        max_seq_len = inputs.shape[1] - 1

        for k in range(0, K):

            list_of_indices = mask_indices_by_span_len[k]
            if list_of_indices.shape == (0,):
                continue
            else:
                for j in range(k+1):
                    max_indices = torch.full((list_of_indices.shape[0],), max_seq_len, dtype=torch.long)
                    left, right = (list_of_indices[:, 0], \
                                   torch.min(list_of_indices[:, 1] + j, max_indices))

                    accum_indices[0].append(left)
                    accum_indices[1].append(right)

        accum_indices[0] = list(filter(lambda x: x.shape != (0,), accum_indices[0]))
        accum_indices[1] = list(filter(lambda x: x.shape != (0,), accum_indices[1]))
        if len(accum_indices[0]) != 0: 
            accum_indices_flatten  = (torch.cat(accum_indices[0]), torch.cat(accum_indices[1]))
            labels_to_be_mask.index_put_(accum_indices_flatten, torch.tensor([1.]).bool())
            labels_to_be_mask.masked_fill_(special_tokens_mask, value=0.0).bool()

        inputs[labels_to_be_mask] = self.tokenizer.mask_token_id
        labels[~labels_to_be_mask] = -100  # We only compute loss on masked token
        
        return inputs, labels