import os
from os.path import join
import logging

import torch
from torch.utils.data.dataset import TensorDataset
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator, Type

from transformers import AutoTokenizer
from tqdm import tqdm

import pickle

logger = logging.getLogger(__name__)

LABEL_PAD_ID = -1

class InputExample(object) :
    def __init__(self, words, labels) :
        self.words = words
        self.labels = labels

class InputFeature(object) :
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids) :
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

class DataProcessor() :
    # Must implement for Custom Dataset
    def __init__(self, hparams) :
        self.hparams = hparams
        # self.file_path = self.get_file(hparams.data_dir, lang, split)
        # self.lang = lang
        # self.split = split
        self.data_dir = hparams.data_dir
        self.pretrain_model = hparams.pretrain_model
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pretrain_model)
        if hparams.max_len is not None :
            assert 0 < hparams.max_len
        self.max_len = min(hparams.max_len, self.tokenizer.model_max_length)
        self.shift = self.max_len // 2
        
        self.labels = self.get_labels()
        self.num_labels = len(self.labels)
        self.label2id = {label:idx for idx, label in enumerate(self.labels)}
        self.id2label = {v:k for k,v in self.label2id.items()}
        self.padding = {
            "sent": self.tokenizer.pad_token_id,
            "labels": LABEL_PAD_ID,
            "lang": 0,
        }
    
    def tokenize(self, token) :
        subwords = self.tokenizer.tokenize(token)
        return subwords
        
    def get_labels(self) :
        return [
            'B-ORG', 'B-DATE', 'B-PERSON', 'B-GPE', 'B-MONEY', 'B-CARDINAL',
            'B-PERCENT', 'B-NORP', 'B-TIME', 'B-WORK_OF_ART', 'B-LOC', 'B-QUANTITY',
            'B-EVENT', 'B-PRODUCT', 'B-FAC', 'B-ORDINAL', 'B-LAW', 'B-LANGUAGE',
            'I-ORG', 'I-DATE', 'I-PERSON', 'I-GPE', 'I-MONEY', 'I-CARDINAL',
            'I-PERCENT', 'I-NORP', 'I-TIME', 'I-WORK_OF_ART', 'I-LOC', 'I-QUANTITY',
            'I-EVENT', 'I-PRODUCT', 'I-FAC', 'I-ORDINAL', 'I-LAW', 'I-LANGUAGE',
            "O",
        ]
    
    # return data in a model-feedable format
    def get_data(self, data_dir, lang, split, only_data = True) : #! only_data
        cache_path = self.get_cache_path(data_dir, lang, split)
        if os.path.exists(cache_path) :
            with open(cache_path, 'rb') as f :
                processed_data = pickle.load(f)
            logger.info(f'===Loaded processed_data for {lang}-{split} from cache path {cache_path}')
            return processed_data
        else:
            logger.info(f'===Creating processed_data for {lang}-{split} and caching it to path {cache_path}')
            file_path = self.get_file(data_dir, lang, split)
            processed_data = self.get_data_helper(file_path, cache_path, lang, split) #!
            return processed_data
            
    def get_cache_path(self, data_dir, lang, split) : # consider model type and max_len also
        cache_dir = join(data_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f'{lang}_{split}_{self.hparams.pretrain_model}_{self.max_len}.pickle'
        cache_path = join(cache_dir, cache_file)
        return cache_path
    
    # return file_path to process    
    def get_file(self, data_dir: str, lang: str, split: str) -> Optional[str] :
        if split == 'train' :
            file_path = f'{data_dir}/{lang}/train.iob2.txt'
        elif split == 'dev' :
            file_path = f'{data_dir}/{lang}/dev.iob2.txt'
        elif split == 'test' :
            file_path = f'{data_dir}/{lang}/test.iob2.txt'
        else :
            raise ValueError(f'Unsupported split: {split}')
        return file_path
    
    def get_data_helper(self, file_path: str, cache_path, lang, split):
        # process data
        examples = []
        logger.info(f'===Creating Examples for {lang}-{split}...')
        examples =  self.read_examples_from_file(file_path)
        logger.info(f'Total N : {len(examples)} & First example : {examples[0]}')
        
        logger.info(f'===Processing examples to trainable instances..')
        processed_data = self.process_examples(examples)
        logger.info(f'Total N : {len(processed_data)} & First instance : {processed_data[0]}')
        
        # cache it
        with open(cache_path, 'wb') as f :
            pickle.dump(processed_data, f, protocol = 4)
        logger.info(f'===Saved Processed data to cache dir')
        
        return processed_data
    
    # \t으로 word와 label이 구분지어진 파일의 경로가 주어지면 -> 문장 단위로 dictionary를 만들어서 yield
    def read_examples_from_file(self, file_path: str) -> Iterator[Dict[str, List]] :
        examples = []
        
        words: List[str] = []
        labels: List[str] = []
        
        with open(file_path, 'r', encoding = 'utf-8') as f :
            for line in f.readlines() :
                line = line.strip()
                if not line :
                    assert words
                    assert len(words) == len(labels)
                    examples.append(InputExample(words = words, labels = labels))
                    words, labels = [], []
                else :
                    word, label = line.split('\t')
                    words.append(word)
                    labels.append(label)
            if words :
                assert len(words) == len(labels)
                examples.append(InputExample(words = words, labels = labels))
        return examples
    
    def process_examples(self, examples) :
        max_num_tokens = self.max_len - (self.tokenizer.model_max_length - self.tokenizer.max_len_single_sentence) # 128 - (512 - 510) = 126
        assert self.tokenizer.padding_side == 'right'
        
        features = []
        for example in examples :
            words = example.words
            labels = example.labels
            
            tokens = []
            label_ids = []
            for word, label in zip(words, labels) :
                assert word
                sub_tokens = self.tokenize(word)
                
                if len(tokens) + len(label_ids) > max_num_tokens :
                    features.append(self.process_example(tokens, label_ids))
                    
                    tokens = tokens[-self.shift :] # 토큰의 반은 남김
                    label_ids = [LABEL_PAD_ID] * len(tokens) # 하지만 남긴 토큰은 LABEL_PAD_ID로 패딩하여 train 혹은 predict하지 않음
                else :
                    tokens += sub_tokens
                    sub_label_ids = [self.label2id[label]] + [LABEL_PAD_ID] * (len(sub_tokens) - 1)
                    label_ids += sub_label_ids
            if tokens :
                features.append(self.process_example(tokens, label_ids))
        
        all_input_ids = torch.LongTensor([f.input_ids for f in features])
        all_attention_mask = torch.LongTensor([f.attention_mask for f in features])
        all_token_type_ids = torch.LongTensor([f.token_type_ids for f in features])
        all_label_ids = torch.LongTensor([f.label_ids for f in features])
        
        return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
                    
    def process_example(self, tokens, label_ids_) :
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        pad_token_id = self.tokenizer.pad_token_id
        pad_token_type_id = self.tokenizer.pad_token_type_id
        
        input_ids = self.tokenizer.convert_tokens_to_ids([cls_token] + tokens + [sep_token])
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        pad_len = self.max_len - len(input_ids)
        
        input_ids += [pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        token_type_ids += [pad_token_type_id] * pad_len
        
        label_ids = [LABEL_PAD_ID] + label_ids_ + [LABEL_PAD_ID] + [LABEL_PAD_ID] * pad_len # cls, sep, pad
        
        if len(input_ids) != len(attention_mask) :
            logger.info(f'{len(input_ids)} != {len(attention_mask)}')
        
        return InputFeature(input_ids, attention_mask, token_type_ids, label_ids)
                
    
    # 문장(단어 리스트)과 라벨들(라벨 리스트)가 들어오면 tokenize->to_ids, special token, label padding까지 끝내서 문장(token id의 array)과 라벨들(label id의 array) return
    def process_example_helper(self, sent: List, labels : List) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
        
        token_ids: List[int] = []
        label_ids: List[int] = []
        
        for token, label in zip(sent, labels) :
            sub_tokens = self.tokenize(token)
            if not sub_tokens :
                continue
            sub_tokens = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            
            if len(token_ids) + len(sub_tokens) >= self.max_len :
                # don't add more token
                yield self.add_special_tokens(token_ids, label_ids) #!
                
                token_ids = token_ids[-self.shift :] # 토큰의 반은 남김
                label_ids = [LABEL_PAD_ID] * len(token_ids) # 하지만 남긴 토큰은 LABEL_PAD_ID로 패딩하여 train 혹은 predict하지 않음
                
            for i, sub_token in enumerate(sub_tokens) :
                token_ids.append(sub_token)
                label_id = self.label2id[label] if i == 0 else LABEL_PAD_ID # 첫번째 sub_token에만 label id를 부여, 나머지 sub_token에는 -1 부여
                label_ids.append(label_id)
        yield self.add_special_tokens(token_ids, label_ids) #!
    
    ##!
    def add_special_tokens(self, sent, labels) :
        sent = self.tokenizer.build_inputs_with_special_tokens(sent)
        labels = self.tokenizer.build_inputs_with_special_tokens(labels)
        mask = self.tokenizer.get_special_tokens_mask( # returns a list of integers in the range [0,1] : 1 for a special token, 0 for a sequence token
            sent, already_has_special_tokens=True
        )
        sent, labels, mask = np.array(sent), np.array(labels), np.array(mask)
        label = labels * (1 - mask) + LABEL_PAD_ID * mask #!스페셜 토큰 자리에 -1
        return sent, label
                
    
if __name__ == "__main__" :
    pass