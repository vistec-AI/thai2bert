'''
Proposed :


from thai2transformers.pipeline.finetuning.sequence_classification import (
    SequenceClassificationFinetuningPipeline
)

or from thai2transformers.pipeline.finetuning import (
    SequenceClassificationFinetuningPipeline
)

seq_cls_finetune = SequenceClassificationFinetuningPipeline()
seq_cls_finetune.load_dataset('wongnai_reviews')
'''
import os
import sys
from typing import List, Union, Dict, Callable, Optional

from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    TrainingArguments
)

from .base import BaseFinetuningPipeline
from thai2transformers.conf import Task
from thai2transformers.finetuner import BaseFinetuner, SequenceClassificationFinetuner
from thai2transformers.datasets import SequenceClassificationDataset
from thai2transformers.utils import get_dict_val


class SequenceClassificationFinetuningPipeline(BaseFinetuningPipeline):

    def __init__(self,
                 task: Union[str, Task],
                 train_dataset: SequenceClassificationDataset = None,
                 val_dataset: SequenceClassificationDataset = None,
                 test_dataset: SequenceClassificationDataset = None,
                 finetuner: BaseFinetuner = None):
        if isinstance(task, Task):
            self.task = task.value
        elif isinstance(task, str):
            self.task = task
        self._dataset = None
        self.tokeinzer = tokeinzer
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.finetuner = finetuner
        self.label_encoder = None

        self.finetuner = SequenceClassificationFinetuner()

    def load_dataset(self,
                     dataset_name_or_path: Union[str, os.PathLike],
                     text_column_name: str,
                     label_column_name: Union[str, List[str]],
                     task: Union[str, Task]= None):
        if isinstance(task, Task):
            self.task = task.value
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        if isinstance(dataset_name_or_path, str):
            self._dataset = load_dataset(dataset_name_or_path)


        if self.task == Task.MULTICLASS_CLS.value:
            if 'train' in self._dataset.keys():
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(self._dataset['train'][label_column_name])
        elif self.task == Task.MULTILABEL_CLS.value:
            self.label_encoder = None
        else:
            raise ValueError()

        # get number of labels
        if self.task == Task.MULTICLASS_CLS.value:
            self.num_labels = len(set(get_dict_val(self._dataset['train'], label_column_name)))
        if self.task == Task.MULTILABEL_CLS.value:
            self.num_labels = len(set(reduce(lambda a,b: a + b, get_dict_val(self._dataset['train'], label_column_name))))


    def load_tokenizer(self, tokenizer_cls, name_or_path):

        self.finetuner.load_pretrained_tokenizer(
                        tokenizer_cls=tokenizer_cls,
                        name_or_path=name_or_path)
   
    def load_model(self, tokenizer_cls, name_or_path, num_labels:int = None):

        if self.num_labels == None and num_labels != None:
            self.num_labels = num_labels

        self.finetuner.load_pretrained_model(
                        task=self.task,
                        name_or_path=name_or_path,
                        num_labels=self.num_labels)

    def process_dataset(self,
                           tokenizer=None,
                           preprocessor: Callable[[str], str]=None,
                           max_length=512,
                           bs=1000,
                           space_token='<_>',
                           train_dataset_name='train',
                           val_dataset_name='val',
                           test_dataset_name='test'):   
 
        if self.tokenizer == None and tokenizer != None:
            self.tokenizer = tokenizer
        elif self.tokenizer == None and tokenizer == None:
            raise AssertionError('A Tokenizer has never been specified')


        self.train_dataset = SequenceClassificationDataset.from_dataset(
            task=self.task,
            tokenizer=self.tokenizer,
            dataset=self._dataset[train_dataset_name],
            text_column_name=self.text_column_name,
            label_column_name=self.label_column_name,
            max_length=max_length,
            bs=bs,
            space_token=space_token,
            preprocessor=preprocessor,
            label_encoder=self.label_encoder,
        )

        if val_dataset_name in self._dataset.keys():
            self.val_dataset = SequenceClassificationDataset.from_dataset(
                task=self.task,
                tokenizer=self.tokenizer,
                dataset=self._dataset[val_dataset_name],
                text_column_name=self.text_column_name,
                label_column_name=self.label_column_name,
                max_length=max_length,
                bs=bs,
                space_token=space_token,
                preprocessor=preprocessor,
                label_encoder=self.label_encoder,
            )
        
        if test_dataset_name in self._dataset.keys():
            self.test_dataset = SequenceClassificationDataset.from_dataset(
                task=self.task,
                tokenizer=self.tokenizer,
                dataset=self._dataset[val_dataset_name],
                text_column_name=self.text_column_name,
                label_column_name=self.label_column_name,
                max_length=max_length,
                bs=bs,
                space_token=space_token,
                preprocessor=preprocessor,
                label_encoder=self.label_encoder,
            )

    def finetune(self, output_dir:str, eval_on_test_set:bool = False, **kwargs):

        training_args = TrainingArguments(output_dir=output_dir,
                                          **kwargs)

        if eval_on_test_set and self.test_dataset:
            return self.finetuner.finetune(training_args,
                                train_dataset=self.train_dataset,
                                val_dataset=self.val_dataset,
                                test_dataset=self.test_dataset)
        elif eval_on_test_set and self.test_dataset == None:
            raise AssertionError('test_dataset is not specified, while argument eval_on_test_set is True')
        
        if not eval_on_test_set:
            self.finetuner.finetune(training_args,
                                train_dataset=self.train_dataset,
                                val_dataset=self.val_dataset,
                                test_dataset=None)