import torch

from typing import Callable, List, Tuple, Union
from functools import partial
import itertools

from seqeval.scheme import Tokens, IOB2, IOBES

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
from pythainlp.tokenize import word_tokenize as pythainlp_word_tokenize
newmm_word_tokenizer = partial(pythainlp_word_tokenize, keep_whitespace=True, engine='newmm')

from thai2transformers.preprocess import rm_useless_spaces

SPIECE = '▁'

class TokenClassificationPipeline:

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 pretokenizer: Callable[[str], List[str]] = newmm_word_tokenizer,
                 lowercase=False,
                 space_token='<_>',
                 device: int = -1,
                 group_entities: bool = False,
                 strict: bool = False,
                 tag_delimiter: str = '-',
                 scheme: str = 'IOB',
                 use_crf=False):

        super().__init__()

        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        assert isinstance(model, PreTrainedModel)
        
        self.model = model
        self.tokenizer = tokenizer
        self.pretokenizer = pretokenizer
        self.lowercase = lowercase
        self.space_token = space_token
        self.device = 'cpu' if device == -1 or not torch.cuda.is_available() else f'cuda:{device}'
        self.group_entities = group_entities
        self.strict = strict
        self.tag_delimiter = tag_delimiter
        self.scheme = scheme
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.use_crf = use_crf
        self.model.to(self.device)

    def preprocess(self, inputs: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:

        if self.lowercase:
            inputs = inputs.lower() if type(inputs) == str else list(map(str.lower, inputs))
        
        inputs = rm_useless_spaces(inputs) if type(inputs) == str else list(map(rm_useless_spaces, inputs))

        tokens = self.pretokenizer(inputs) if type(inputs) == str else list(map(self.pretokenizer, inputs))

        tokens = list(map(lambda x: x.replace(' ', self.space_token), tokens)) if type(inputs) == str else \
                 list(map(lambda _tokens: list(map(lambda x: x.replace(' ', self.space_token), _tokens)), tokens))

        return tokens

    def _inference(self, input: str):

        tokens = [[self.tokenizer.bos_token]] + \
                    [self.tokenizer.tokenize(tok) if tok != SPIECE else [SPIECE] for tok in self.preprocess(input)] + \
                    [[self.tokenizer.eos_token]]
        ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        flatten_tokens = list(itertools.chain(*tokens))
        flatten_ids = list(itertools.chain(*ids))

        input_ids = torch.LongTensor([flatten_ids]).to(self.device)

        if self.use_crf:
            out = self.model(input_ids=input_ids)
        else:
            out = self.model(input_ids=input_ids, return_dict=True)
            probs = torch.softmax(out['logits'], dim=-1)
            vals, indices = probs.topk(1)
            indices_np = indices.detach().cpu().numpy().reshape(-1)

        list_of_token_label_tuple = list(zip(flatten_tokens, [ self.id2label[idx] for idx in indices_np] ))
        merged_preds = self._merged_pred(preds=list_of_token_label_tuple, ids=ids)
        merged_preds_removed_spiece = list(map(lambda x: (x[0].replace(SPIECE, ''), x[1]), merged_preds))

        # remove start and end tokens
        merged_preds_removed_bos_eos = merged_preds_removed_spiece[1:-1]
        # convert to list of Dict objects
        merged_preds_return_dict = [ {'word': word if word != self.space_token else ' ', 'entity': tag, 'index': idx } \
            for idx, (word, tag) in enumerate(merged_preds_removed_bos_eos) ]

        if (not self.group_entities or self.scheme == None) and self.strict == True:
            return merged_preds_return_dict
        elif not self.group_entities and self.strict == False:

            tags = list(map(lambda x: x['entity'], merged_preds_return_dict))
            processed_tags = self._fix_incorrect_tags(tags)
            for i, item in enumerate(merged_preds_return_dict):
                merged_preds_return_dict[i]['entity'] = processed_tags[i]
            return merged_preds_return_dict
        elif self.group_entities:
            return self._group_entities(merged_preds_removed_bos_eos)

    def __call__(self, inputs: Union[str, List[str]]):

        """     
            
        """
        if type(inputs) == str:
            return self._inference(inputs)
        
        if type(inputs) == list:
            results = [ self._inference(text) for text in inputs]
            return results
       

    def _merged_pred(self, preds: List[Tuple[str, str]], ids: List[List[int]]):
    
        token_mapping = [ ]
        for i in range(0, len(ids)):
            for j in range(0, len(ids[i])):
                token_mapping.append(i)

        grouped_subtokens = []
        _subtoken = []
        prev_idx = 0
    
        for i, (subtoken, label) in enumerate(preds):
            
            current_idx =  token_mapping[i]
            if prev_idx != current_idx:
                grouped_subtokens.append(_subtoken)
                _subtoken = [(subtoken, label)]
                if i == len(preds) -1:
                    _subtoken = [(subtoken, label)]
                    grouped_subtokens.append(_subtoken)
            elif i == len(preds) -1:
                _subtoken += [(subtoken, label)]
                grouped_subtokens.append(_subtoken)
            else:
                _subtoken += [(subtoken, label)]
            prev_idx = current_idx
        
        merged_subtokens = []
        _merged_subtoken = ''
        for subtoken_group in grouped_subtokens:
            
            first_token_pred = subtoken_group[0][1]
            _merged_subtoken = ''.join(list(map(lambda x: x[0], subtoken_group)))
            merged_subtokens.append((_merged_subtoken, first_token_pred))
        return merged_subtokens

    def _fix_incorrect_tags(self, tags: List[str]) -> List[str]:

        I_PREFIX = f'I{self.tag_delimiter}'
        E_PREFIX = f'E{self.tag_delimiter}'
        B_PREFIX = f'B{self.tag_delimiter}'
        O_PREFIX = 'O'
    
        previous_tag_ne = None
        for i, current_tag in enumerate(tags):
            
            current_tag_ne = current_tag.split(self.tag_delimiter)[-1] if current_tag != O_PREFIX else O_PREFIX
            
            if i == 0 and (current_tag.startswith(I_PREFIX) or \
                current_tag.startswith(E_PREFIX)):
                # if a NE tag (with I-, or E- prefix) occuring at the begining of sentence
                # e.g. (I-LOC, I-LOC) , (E-LOC, B-PER) (I-LOC, O, O)
                # then, change the prefix of the current tag to B{tag_delimiter}
                tags[i] = B_PREFIX + tags[i][2:]
            elif i >= 1 and tags[i-1] == O_PREFIX and (
                current_tag.startswith(I_PREFIX) or \
                current_tag.startswith(E_PREFIX)):
                # if a NE tag (with I-, or E- prefix) occuring after O tag
                # e.g. (O, I-LOC, I-LOC) , (O, E-LOC, B-PER) (O, I-LOC, O, O)
                # then, change the prefix of the current tag to B{tag_delimiter}
                tags[i] = B_PREFIX + tags[i][2:]
            elif i >= 1 and ( tags[i-1].startswith(I_PREFIX) or \
                tags[i-1].startswith(E_PREFIX) or \
                tags[i-1].startswith(B_PREFIX)) and \
                ( current_tag.startswith(I_PREFIX) or current_tag.startswith(E_PREFIX) )  and \
                previous_tag_ne != current_tag_ne:
                # if a NE tag (with I-, or E- prefix) occuring after NE tag with different NE
                # e.g. (B-LOC, I-PER) , (B-LOC, E-LOC, E-PER) (B-LOC, I-LOC, I-PER)
                # then, change the prefix of the current tag to B{tag_delimiter}
                tags[i] = B_PREFIX + tags[i][2:]
            elif i == len(tags) - 1 and tags[i-1] == O_PREFIX and (
                current_tag.startswith(I_PREFIX) or \
                current_tag.startswith(E_PREFIX)):
                # if a NE tag (with I-, or E- prefix) occuring at the end of sentence
                # e.g. (O, O, I-LOC)  , (O, O, E-LOC) 
                # then, change the prefix of the current tag to B{tag_delimiter}
                tags[i] = B_PREFIX + tags[i][2:]

            previous_tag_ne = current_tag_ne
        
        return tags

    def _group_entities(self, ner_tags: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        
        if self.scheme not in ['IOB', 'IOBES', 'IOBE']:
            raise AttributeError()

        tokens, tags = zip(*ner_tags)
        tokens, tags = list(tokens), list(tags)

        if self.scheme == 'IOBE':
            # Replace E prefix with I prefix
            tags = list(map(lambda x: x.replace(f'E{self.tag_delimiter}', f'I{self.tag_delimiter}'), tags))
        if self.scheme == 'IOBES':
            # Replace E prefix with I prefix and replace S prefix with B
            tags = list(map(lambda x: x.replace(f'E{self.tag_delimiter}', f'I{self.tag_delimiter}'), tags))
            tags = list(map(lambda x: x.replace(f'S{self.tag_delimiter}', f'B{self.tag_delimiter}'), tags))

        if not self.strict:
            
            tags = self._fix_incorrect_tags(tags)
            
        ent = Tokens(tokens=tags, scheme=IOB2,
                     suffix=False, delimiter=self.tag_delimiter)

        ne_position_mappings = ent.entities

        num_tags = len(tags)
        begin_end_pos = []
        for i, ne_position_mapping in enumerate(ne_position_mappings):
            begin_end_pos.append((ne_position_mapping.start, ne_position_mapping.end))
        
        j = 0
        # print(f'tokens: {tokens}')
        for i, pos_tuple in enumerate(begin_end_pos):   
            # print(f'j = {j}')
            if pos_tuple[0] > 0 and i == 0:
                ne_position_mappings.insert(0, (None, 'O', 0, pos_tuple[0]))
                j += 1   
            if begin_end_pos[i-1][1] != begin_end_pos[i][0] and len(begin_end_pos) > 1 and i > 0 :
                ne_position_mappings.insert(j, (None, 'O', begin_end_pos[i-1][1], begin_end_pos[i][0]))
                j += 1 
            if begin_end_pos[i][1] != num_tags and i == len(begin_end_pos) - 1:
                # print(f' c3 j={j}, begin_end_pos[i][1]={begin_end_pos[i][1]}, num_tags={num_tags}')
                ne_position_mappings.insert(j+1, (None, 'O', begin_end_pos[i][1], num_tags))
                
            j += 1
            # print('ne_position_mappings', ne_position_mappings) 

        groups = []
        for i, ne_position_mapping in enumerate(ne_position_mappings):
            if type(ne_position_mapping) != tuple:
                ne_position_mapping = ne_position_mapping.to_tuple()
            ne = ne_position_mapping[1]
            text = ''
            for ne_position in range(ne_position_mapping[2], ne_position_mapping[3]):
                _token = tokens[ne_position]
                text += _token if _token != self.space_token else ' '
            groups.append({
                'entity_group': ne,
                'word': text
            })
    
        return groups
